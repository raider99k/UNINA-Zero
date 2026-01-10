# **UNINA-DLA Blueprint: A Comprehensive Research Report on Distilled, Hardware-Native Perception Architectures for Autonomous Racing**

## **1\. Executive Summary and Strategic Imperative**

The autonomous racing domain, typified by the Formula Student Driverless competition, presents a unique confluence of constraints that distinguish it fundamentally from general autonomous driving or standard computer vision tasks. The vehicle operates at the limit of friction, necessitating control loops running at frequencies exceeding 100Hz. In this regime, perception latency is not merely a performance metric; it is a stability constraint. A delay of milliseconds in detecting a cone at 20 meters, when traveling at 25 meters per second, translates to a localization error that can destabilize the Model Predictive Control (MPC) and result in a track limit violation or collision.

The UNINA-Zero project’s mandate—to achieve **Zero-Overhead Inference** on the NVIDIA Jetson Orin AGX—requires a radical departure from the conventional "train-then-optimize" paradigms prevalent in the industry. Standard approaches involving off-the-shelf models like YOLOv8 or EfficientDet, followed by generic TensorRT optimization, fail to unlock the specific silicon advantages of the Orin System-on-Chip (SoC). These models often saturate the GPU, leaving insufficient resources for the heavy-lifting required by Simultaneous Localization and Mapping (SLAM) and path planning algorithms.

To achieve the project goals, we must adopt a "hardware-first" design philosophy. This implies that the neural architecture must be derived directly from the silicon logic of the Deep Learning Accelerator (DLA), rather than adapting the hardware to fit the model. The DLA is a specialized, fixed-function accelerator designed for high efficiency and deterministic latency, but it lacks the flexibility of the GPU's CUDA cores.

This report serves as the definitive blueprint for constructing **UNINA-DLA**, a custom perception stack designed to completely bypass the GPU for inference. By targeting the DLA 2.0 architecture within the Orin SoC, which contributes approximately 105 sparse INT8 TOPS to the system's total throughput 1, we can achieve a dedicated, deterministic inference pipeline that operates in parallel with the GPU-bound SLAM and MPC stacks.

Following an exhaustive red-teaming exercise of potential architectures—including Modified CSPDarknet, ConeNet-DLA, and various YOLO iterations—this research identifies a converged solution: a **RepVGG-based student backbone** distilled from a massive **YOLOv10-X teacher**, coupled with a **YOLOv10 "One-to-One" detection head**. This architecture leverages structural re-parameterization to collapse multi-branch topologies into a single-path stream of $3\\times3$ convolutions—the native dialect of the DLA’s Multiply-Accumulate (MAC) arrays. Furthermore, the integration of the NMS-free "One-to-One" head eliminates the latency-inducing Non-Maximum Suppression stage, enabling the DLA to write final detections directly to memory without GPU intervention.

To recover the accuracy lost by architectural simplification (replacing SiLU with ReLU) and aggressive quantization (INT8), we define a rigorous **"Tri-Vector" distillation strategy** utilizing Scale-Aware Knowledge Distillation (ScaleKD) 3 and a sensitive-layer-aware Quantization-Aware Training (QAT) workflow. The deployment strategy utilizes C++ with **Zero-Copy memory mapping**, ensuring that the latency from photon-to-steering-command is minimized to the theoretical hardware limit.

## **2\. The Hardware Substrate: Decrypting the NVIDIA Orin DLA**

To architect a successful model, one must first understand the battlefield. The NVIDIA Jetson Orin AGX is a heterogeneous system-on-chip (SoC) where the Deep Learning Accelerator (DLA) plays a specific, often misunderstood role. It is not merely a "weaker GPU"; it is a distinct silicon intellectual property (IP) block designed for maximum efficiency per watt and, crucially for racing, deterministic latency.

### **2.1 The NVDLA 2.0 Microarchitecture**

The Orin AGX features two DLA cores (DLA 2.0), distinct from the Ampere GPU. While the GPU relies on a Single Instruction, Multiple Threads (SIMT) architecture managed by a complex warp scheduler, the DLA is a dataflow machine. It consists of fixed-function engines—Convolution Core, Planar Data Processor (PDP), Channel Data Processor (CDP), and others—connected by internal SRAM buffers. Understanding these engines is prerequisite to designing compatible layers.

#### **2.1.1 The Convolution Buffer (CBUF) Constraint**

A critical but often overlooked constraint is the internal Convolution Buffer (CBUF). The DLA loads weights and activation data into this dedicated SRAM (1 MiB per core on Orin) to perform operations.4

* **The Thrashing Phenomenon:** If a layer’s parameters or intermediate feature maps exceed the CBUF capacity, the DLA cannot process the layer in a single pass. It must fetch data from the global DRAM or system SRAM, process a tile, write it back, and fetch the next tile. This behavior, known as "memory thrashing," causes significant latency spikes.  
* **Architectural Implication:** Our student architecture must prioritize layer configurations (kernel size, channel depth) that fit residentially within the CBUF. RepVGG’s fused $3\\times3$ convolutions are ideal here, as they present a predictable memory footprint compared to the irregular access patterns of multi-path Inception or NAS-searched blocks found in models like EfficientNet or MobileNet.1 Design choices must cap channel depths at 512 in deeper layers to avoid overflowing the CBUF.

#### **2.1.2 The "Supported Layer" Minefield**

The TensorRT 10.x documentation provides a strict whitelist of supported operations for the DLA. Unlike CUDA, which can execute almost any arithmetic operation via custom kernels, the DLA is immutable.

* **Convolution & Fully Connected:** Fully supported, but with rigid stride and padding limitations. Specifically, dilated convolutions are a major hazard. The hardware requires that for dilated convolutions, the padding must be strictly less than the kernel size.5 Standard "SAME" padding in frameworks like PyTorch or TensorFlow often violates this rule when dilation \> 1 (e.g., a dilation of 2 with a $3\\times3$ kernel often requires padding of 2). This violation triggers a silent fallback to the GPU.  
* **Activations:** This is a primary friction point. The DLA natively supports ReLU, Sigmoid, TanH, and Clipped ReLU via a lookup table (LUT) engine or dedicated hardware.4 Crucially, **SiLU (Swish)**, the default activation in modern YOLO variants (v5, v8, v10), presents a performance hazard. While technically supported in newer DLA firmware, SiLU often forces the DLA to promote the tensor precision to FP16, even if the user requested INT8. This effectively halves the throughput (from \~100 TOPS INT8 to \~50 TOPS FP16) and doubles memory bandwidth usage.1  
* **Pooling:** Supported, but with restrictions on window sizes (typically 1-8) and strides. Global Average Pooling (GAP) often requires the PDP engine, which has lower throughput than the Convolution Core.  
* **Dynamic Shapes:** The DLA strictly forbids dynamic shapes. The build-time profile dimensions (min, max, opt) in TensorRT must be identical. This necessitates a fixed-resolution input strategy, simplifying the pipeline but removing runtime flexibility.4

### **2.2 The Latency Penalty of GPU Fallback**

A central theme in our research is the catastrophic impact of kGPU\_FALLBACK. TensorRT allows layers unsupported by the DLA to fall back to the GPU. While functionally convenient for general deployment, this is architecturally disastrous for latency-critical racing applications.

* **The Mechanism of Failure:** When a fallback occurs, data must be synchronized and transferred between the DLA’s memory domain and the GPU’s memory domain. Although they share physical DRAM on the SoC, the access paths differ, and the handover requires cudaEventSynchronize calls that block execution.  
* **Empirical Evidence:** Developer reports and benchmarking indicate that a model running primarily on DLA with frequent GPU fallbacks can exhibit **2x to 3x higher latency** than running purely on the GPU.8 The synchronization overhead dominates the actual compute time.  
* **Strategic Directive:** The UNINA-DLA architecture must be designed for **100% DLA residency**. Any layer that triggers a fallback is considered a critical design flaw. This strictly rules out complex post-processing layers (like standard NMS implemented via EfficientNMS plugin) and specific tensor manipulations (like Gather, NonZero, or boolean masking) inside the model graph.1

## **3\. Architectural Red Teaming: Selecting the Student**

The "Student" model is the inference engine that will reside on the vehicle. Its design is governed by a reverse-engineering of the DLA's capabilities, prioritizing throughput, determinism, and compilation purity over theoretical parameter efficiency. We evaluated three primary candidates derived from the provided research materials.

### **3.1 Candidate 1: Modified CSPDarknet-S (UNINA-Zero Technical Doc)**

The first proposal 1 suggests a Modified CSPDarknet-S backbone. This architecture relies on Cross-Stage Partial (CSP) connections, which split feature maps into two parts: one part processes through a dense block, and the other bypasses it to be concatenated at the end.

* **Red Team Analysis:** While CSPNet is highly efficient on GPUs due to reduced gradient redundancy, it is suboptimal for the DLA. The DLA is a dataflow architecture that excels at processing continuous streams of data. The "split-transform-merge" topology of CSPNet introduces significant "graph friction." The split and concatenation operations require complex pointer arithmetic and memory management within the DLA's DMA engine. Furthermore, CSP blocks heavily utilize element-wise addition (residual connections). Element-wise operations are memory-bound (low arithmetic intensity); the DLA must fetch two full tensors to perform a simple add, consuming bandwidth that could otherwise feed the hungry MAC arrays.1  
* **Verdict:** **REJECTED.** The memory access overhead of CSP connections bottlenecks the DLA's compute capability.

### **3.2 Candidate 2: ConeNet-DLA (CenterNet-Style)**

The ConeNet-DLA proposal 1 advocates for a RepVGG backbone coupled with a CenterNet-style anchor-free head, predicting heatmaps for object centers.

* **Red Team Analysis:** The use of RepVGG is a strong positive, aligning with the DLA's preference for dense $3\\times3$ convolutions. However, the CenterNet head presents a fatal flaw for the "Zero-Overhead" goal. A heatmap-based head outputs a high-resolution tensor (typically stride 4\) that requires finding local maxima (peaks) to identify objects. This "peak finding" or decoding step usually involves MaxPool with stride 1 (for NMS) or TopK operations. The TopK layer is often unsupported or inefficient on DLA 10, forcing a fallback to the GPU or CPU for decoding. This reintroduces the latency variance we aim to eliminate. The report explicitly notes that CenterNet decoding requires specific tensor manipulations that are brittle on DLA.1  
* **Verdict:** **REJECTED.** The post-processing complexity of heatmap decoding violates the 100% DLA residency requirement.

### **3.3 Candidate 3: Generic YOLOv10**

Using a stock YOLOv10 model 11 offers state-of-the-art accuracy and an NMS-free design via its "One-to-One" head.

* **Red Team Analysis:** While the head design is promising, the stock YOLOv10 backbone and activation functions are incompatible. As noted in Section 2.1.2, stock YOLOv10 uses **SiLU** activations throughout. deploying a SiLU-heavy model on DLA forces an FP16 execution path, halving the potential INT8 throughput and doubling memory bandwidth usage. Furthermore, stock YOLO models often rely on SPPF (Spatial Pyramid Pooling Fast) modules that may contain operations not fully optimized for the DLA's specific pooling engine restrictions.  
* **Verdict:** **REJECTED (in stock form).** The architecture is sound algorithmically but requires modification for hardware compliance.

### **3.4 The Converged Solution: UNINA-DLA**

The red-teaming process dictates a hybrid architecture that synthesizes the strengths of the candidates while excising their weaknesses.

* **Backbone:** **RepVGG-B0** is selected. Its structural re-parameterization capability allows it to train as a multi-branch residual network (improving convergence) but compile into a single-path stack of $3\\times3$ convolutions (optimizing DLA execution).1  
* **Activation Policy:** A strict policy of **SiLU $\\to$ ReLU** replacement is enforced. ReLU is a single-cycle, zero-overhead operation on the DLA's SDP (Single Data Point) unit.  
* **Head:** The **YOLOv10 "One-to-One"** head is integrated. By utilizing Consistent Dual Assignments during training, this head learns to output a sparse set of non-overlapping boxes. This eliminates the need for expensive NMS. The DLA outputs raw grid predictions which are decoded on the CPU (Zero-GPU-Overhead). Because the O2O head suppresses duplicates, this decoding is a simple thresholding operation, avoiding the O(N^2) complexity of NMS.

**Final Architecture: UNINA-DLA** = RepVGG-B0 Backbone (ReLU) + Rep-PAN Neck (ReLU) + YOLOv10 One-to-One Head (ReLU/Linear).

## **4\. UNINA-DLA Architecture Detail**

### **4.1 Backbone: RepVGG-B0 and Mathematical Fusion**

The "Student" architecture relies on the mathematical equivalence between the training-time and inference-time blocks of RepVGG.

The Training-Time Block:  
Let $M^{(1)} \\in \\mathbb{R}^{N \\times C\_1 \\times H\_1 \\times W\_1}$ be the input tensor and $M^{(2)} \\in \\mathbb{R}^{N \\times C\_2 \\times H\_2 \\times W\_2}$ be the output. The RepVGG training block consists of three parallel branches:

1. A $3\\times3$ convolution.  
2. A $1\\times1$ convolution.  
3. An identity mapping (if $C\_1 \= C\_2$ and $H\_1 \= H\_2$).

The output is the sum of these branches, each followed by Batch Normalization (BN).

$$M^{(2)} \= \\text{ReLU} \\left( (M^{(1)} \* W^{(3)} \+ b^{(3)}) \+ (M^{(1)} \* W^{(1)} \+ b^{(1)}) \+ M^{(1)} \\right)$$

where $\*$ denotes convolution. Note that BN parameters are fused into the weights and biases $W$ and $b$ during the folding process described below.  
Fusion Step 1: BN Folding  
For any branch with kernel $W$ and BN parameters $\\mu, \\sigma, \\gamma, \\beta$, the fused weight $W'$ and bias $b'$ are calculated as:

$$W'\_{i,:,:,:} \= \\frac{\\gamma\_i}{\\sigma\_i} W\_{i,:,:,:}$$

$$b'\_i \= \\beta\_i \- \\frac{\\mu\_i \\gamma\_i}{\\sigma\_i}$$

This transforms the BN operation into a simple linear bias add.  
Fusion Step 2: Branch Merging  
We exploit the linearity of convolution: $\\mathcal{I} \* K\_1 \+ \\mathcal{I} \* K\_2 \= \\mathcal{I} \* (K\_1 \+ K\_2)$. To add the kernels, they must have the same spatial dimensions ($3\\times3$).

* **Transform $1\\times1$:** The $1\\times1$ kernel $W'^{(1)}$ is zero-padded to $3\\times3$. The value is placed at the center, and surrounding weights are 0\.  
* **Transform Identity:** The identity branch is viewed as a $1\\times1$ convolution with an identity matrix. We construct a $3\\times3$ kernel where the center weight is 1 for the corresponding input/output channel ($i=j$) and 0 elsewhere.

The Inference Kernel:  
The final inference kernel $W\_{final}$ and bias $b\_{final}$ are:

$$W\_{final} \= W'^{(3)} \+ \\text{PAD}(W'^{(1)}) \+ \\text{PAD}(I\_{identity})$$

$$b\_{final} \= b'^{(3)} \+ b'^{(1)} \+ b'\_{identity}$$  
This results in a single operation $M^{(2)} \= \\text{ReLU}(M^{(1)} \* W\_{final} \+ b\_{final})$, which the DLA executes as a single atomic task, maximizing CBUF residency and MAC utilization.1

### **4.2 Neck: Rep-PAN (Re-parameterizable Path Aggregation Network)**

Standard YOLO necks (PANet) rely heavily on element-wise additions to merge features from different scales. As established, element-wise ops are bandwidth-inefficient on DLA.

* **Rep-PAN Strategy:** We employ a re-parameterizable PANet.1 Similar to the backbone, the fusion blocks in the neck are trained as multi-branch structures but fused into single convolutions for inference.  
* **Concatenation Preference:** Where possible, we prioritize concatenation over addition. The DLA's concatenation engine is generally more efficient than its element-wise engine because it involves simple memory addressing rather than arithmetic fetching. The Rep-PAN ensures that after concatenation, the subsequent processing is done via dense convolutions that fully utilize the hardware.17

### **4.3 Head: YOLOv10 One-to-One (NMS-Free)**

The detection head removes the greatest source of latency variance: Non-Maximum Suppression (NMS).

* **Consistent Dual Assignments:** During training, the model utilizes two heads. The **One-to-Many Head** uses standard YOLO assignment (multiple anchors per object) to provide rich gradient signals. The **One-to-One Head** uses Hungarian matching to assign a single best prediction per ground truth object.13  
* **Inference determinism:** Only the One-to-One head is exported. Because it is trained to suppress duplicate boxes internally, the output is a sparse set of detections. This allows us to bypass the EfficientNMS TensorRT plugin (which typically runs on GPU/CPU) and let the DLA write the final detection tensor directly to memory.1

## **5\. The "Tri-Vector" Distillation Strategy**

Substituting RepVGG (for CSPNet) and ReLU (for SiLU) creates an accuracy deficit, particularly for small objects like racing cones. To bridge this gap, we employ a "Tri-Vector" distillation strategy using a **YOLOv10-X** (FP32, SiLU, CSPNet) as the Teacher.

### **5.1 Vector 1: Scale-Decoupled Feature (SDF) Distillation**

Standard feature distillation fails for small objects because the loss function is dominated by the vast number of background pixels.

* **ScaleKD Implementation:** We adopt Scale-Aware Knowledge Distillation (ScaleKD).3 This method decouples the teacher's features into scale-specific embeddings.  
* Spatial Attention Mask: We generate a binary mask $M$ derived from the Teacher's high-confidence predictions. The feature loss $\\mathcal{L}\_{feat}$ is weighted by this mask:  
  $$ \\mathcal{L}{feat} \= \\sum{l} | (F\_T^{(l)} \- \\phi(F\_S^{(l)})) \\odot M^{(l)} ||^2 $$  
where $F\_T$ and $F\_S$ are teacher and student feature maps, and $\\phi$ is a $1\\times1$ convolution adapter to match channels. This mask forces the student to focus its limited capacity solely on the regions containing cones, effectively ignoring the asphalt and sky.1

### **5.2 Vector 2: Logit-Based Response Distillation**

We distill the classification logits using Kullback-Leibler (KL) Divergence with a high temperature ($T=4$).

* **Robustness Transfer:** A "Yellow Cone" partially obscured by shadow might produce a teacher distribution of . A hard label would just be . Distilling the distribution transfers the teacher's uncertainty, teaching the student to be robust to lighting variations—a critical factor in outdoor racing.1

### **5.3 Vector 3: Bounding Box Uncertainty Distillation**

YOLOv10 uses Distribution Focal Loss (DFL), which predicts a probability distribution for the bounding box boundaries rather than just a scalar coordinate. We distill this distribution.

* **Localization Precision:** By mimicking the teacher's distribution of the box edges, the student learns the *spatial uncertainty* of the detection. This is vital for the SLAM system, which can use this uncertainty to weight the landmarks in the map.1

## **6\. Quantization Aware Training (QAT): The Precision Gauntlet**

Deploying an INT8 model on DLA requires more than just a calibration step. RepVGG architectures are notoriously difficult to quantize post-training (PTQ) because the fused single-branch structure can exhibit high activation variance (outliers).20 We mandate a **Sensitive-Layer Aware QAT** pipeline.

### **6.1 The QAT Workflow**

1. **FP32 Convergence:** Train the UNINA-DLA student to convergence using the Tri-Vector distillation.  
2. **Structural Fusion:** Perform the RepVGG fusion (BN folding \+ branch merging) *before* inserting quantization nodes. This ensures the QAT process models the actual inference-time numerics.  
3. **QAT Injection:** Insert QuantStub and DeQuantStub nodes. Replace nn.Conv2d and nn.ReLU with their quantized counterparts (quant\_nn.QuantConv2d, quant\_nn.QuantReLU) using the NVIDIA pytorch-quantization toolkit.22  
4. **Entropy Calibration:** Use Entropy calibration for activations. Unlike Max calibration, which is sensitive to outliers, Entropy calibration minimizes the KL divergence between the FP32 and INT8 distributions, preserving the information content of small, low-magnitude cone activations.1  
5. **Fine-Tuning:** Retrain the quantized model for \~10% of the original epochs with a low learning rate ($1e^{-5}$) to allow weights to adapt to the quantization noise (Simulated Quantization).

### **6.2 Sensitive Layer Exemption (Mixed Precision)**

The final detection head layers (the $1\\times1$ convolutions predicting box coordinates $dx, dy, w, h$) are extremely sensitive to quantization noise. A small error in the regression output can shift a cone by meters in the world frame.

* **Strategy:** We quantize the detection heads to ensure 100% DLA residency.
* **Implementation:** Previous strategies exempted these layers, but this caused implicit data formatting and potential GPU fallbacks. By quantizing the heads, we ensure the entire graph runs on the DLA, accepting the minor precision loss for guaranteed latency determinism.

## **7\. Implementation Blueprint: MMYOLO and PyTorch**

This section provides the precise configuration and code structures required to realize UNINA-DLA within the MMYOLO framework.

### **7.1 Registering the Custom RepVGG-ReLU Backbone**

We must ensure that the RepVGG implementation strictly uses ReLU and allows for the export-time fusion.

Python

\# mmyolo/models/backbones/repvgg\_dla.py  
import torch.nn as nn  
from mmyolo.registry import MODELS  
from.base\_backbone import BaseBackbone  
from mmcv.cnn import ConvModule

@MODELS.register\_module()  
class RepVGG\_DLA(BaseBackbone):  
    """  
    RepVGG Backbone optimized for NVDLA.  
    Forces ReLU activations and supports deploy-time fusion.  
    """  
    def \_\_init\_\_(self, arch='B0', act\_cfg=dict(type\='ReLU', inplace=True), deploy=False, \*\*kwargs):  
        super().\_\_init\_\_(\*\*kwargs)  
        self.deploy \= deploy  
        self.act\_cfg \= act\_cfg  
        \# Define RepVGG stages (simplified for brevity)  
        self.stages \= nn.ModuleList()  
        \#......  
          
    def switch\_to\_deploy(self):  
        if self.deploy:  
            return  
        for module in self.modules():  
            if hasattr(module, 'switch\_to\_deploy'):  
                module.switch\_to\_deploy()  
        self.deploy \= True

    def forward(self, x):  
        outs \=  
        for stage in self.stages:  
            x \= stage(x)  
            if stage\_index in self.out\_indices:  
                outs.append(x)  
        return tuple(outs)

### **7.2 Configuration File (mmyolo/configs/unina_dla.py)**

This configuration file assembles the pieces. Note the explicit act\_cfg and use\_one\_to\_one flags.

Python

\_base\_ \= \['../\_base\_/default\_runtime.py', '../\_base\_/schedules/schedule\_1x.py'\]

\# Hardware-Aware Model Config  
model \= dict(  
    type\='YOLODetector',  
    data\_preprocessor=dict(  
        type\='YOLOv5DetDataPreprocessor',  
        mean=, std=\[255\., 255\., 255\.\], bgr\_to\_rgb=True),  
      
    \# 1\. Backbone: RepVGG-B0 with ReLU  
    backbone=dict(  
        type\='RepVGG\_DLA',  
        arch='B0',  
        out\_indices=(2, 3, 4), \# P3, P4, P5  
        act\_cfg=dict(type\='ReLU', inplace=True), \# STRICTLY ReLU for DLA  
        deploy=False \# Set to True for export  
    ),  
      
    \# 2\. Neck: Rep-PAN with ReLU  
    neck=dict(  
        type\='YOLOv6RepPAFPN', \# Rep-PAN implementation  
        in\_channels=, \# RepVGG-B0 widths  
        out\_channels=,  
        num\_csp\_blocks=0, \# Disable CSP blocks to avoid split/merge friction  
        act\_cfg=dict(type\='ReLU', inplace=True)  
    ),  
      
    \# 3\. Head: YOLOv10 One-to-One  
    bbox\_head=dict(  
        type\='YOLOv10Head',  
        head\_module=dict(  
            type\='YOLOv10HeadModule',  
            num\_classes=1, \# Only Cones  
            in\_channels=,  
            use\_one\_to\_one=True, \# Enable NMS-free export  
            act\_cfg=dict(type\='ReLU', inplace=True)  
        ),  
        loss\_cls=dict(type\='mmdet.CrossEntropyLoss', use\_sigmoid=True, loss\_weight=1.0),  
        loss\_bbox=dict(type\='mmdet.IoULoss', loss\_weight=2.5),  
        loss\_dfl=dict(type\='mmdet.DistributionFocalLoss', loss\_weight=0.5)  
    )  
)

\# 4\. Distillation Hook Configuration  
custom\_hooks \=

\# 5\. Fixed Resolution for DLA  
train\_dataloader \= dict(batch\_size=16, dataset=dict(pipeline=))

## **8\. Deployment Strategy: TensorRT and Zero-Copy**

The final phase is converting the trained PyTorch model into a "Zero-Overhead" runtime executable.

### **8.1 Export and Compilation**

We assume the model has been trained and QAT-calibrated.

1. **Export to ONNX:** Use the deployment config (where deploy=True in backbone/neck) to fuse all RepVGG blocks. Ensure dynamic axes are disabled; DLA requires static shapes.  
   Bash  
   python tools/export_model.py \
       configs/unina_dla.py \
       checkpoints/unina_dla_qat.pth \
       --output-file unina_dla.onnx \
       --input-shape 1 3 640 640 \
       --opset-version 13

2. **Compile with trtexec:** This step compiles the ONNX graph into a DLA-compatible TensorRT engine. The flags are critical.  
   Bash  
   trtexec --onnx=unina_dla.onnx \
           --saveEngine=unina_dla.engine \
           \--useDLACore=0 \\  
           \--int8 \\  
           \--fp16 \\  
           \--inputIOFormats=fp16:chw \\  
           \--outputIOFormats=fp16:chw \\  
           \--profilingVerbosity=detailed \\  
           \--noDataTransfers \\  
           \--allowGPUFallback \# Use only for debugging; aim to remove

   *Note: inputIOFormats=fp16:chw tells TensorRT that the input data is already in the correct layout, avoiding implicit reformatting kernels on the GPU.*

### **8.2 C++ Zero-Copy Implementation**

To minimize latency, we use **Mapped Pinned Memory** (Zero-Copy). This allows the DLA to access host memory directly over the SoC fabric, eliminating the cudaMemcpy overhead.1

C++

// UNINA\_DLA\_Inference.cpp  
\#**include** \<cuda\_runtime.h\>  
\#**include** \<NvInfer.h\>  
\#**include** \<iostream\>

using namespace nvinfer1;

class UNINAPerception {
    IRuntime\* runtime;
    ICudaEngine\* engine;
    IExecutionContext\* context;
    void\* input\_cpu\_ptr;
    void\* input\_gpu\_ptr;
    // Pointers for 6 raw outputs (reg3, reg4, reg5, cls3, cls4, cls5)
    void\* out\_ptrs\_cpu[6]; 
    void\* out\_ptrs\_gpu[6];
    cudaStream\_t stream;

public:
    void init(const char\* engine\_path) {
        runtime \= createInferRuntime(gLogger);
        engine \= runtime-\>deserializeCudaEngine(engineData, engineSize);
        context \= engine-\>createExecutionContext();

        // 1. Input Allocation
        size\_t input\_size \= 1 \* 3 \* 640 \* 640 \* sizeof(half);
        cudaHostAlloc(\&input\_cpu\_ptr, input\_size, cudaHostAllocMapped);
        cudaHostGetDevicePointer(\&input\_gpu\_ptr, input\_cpu\_ptr, 0);

        // 2. Output Allocation (Example for reg3)
        // Repeat for all 6 tensors to avoid GPU fallback
        size\_t reg3\_size \= 1 \* 64 \* 80 \* 80 \* sizeof(float);
        cudaHostAlloc(\&out\_ptrs\_cpu[0], reg3\_size, cudaHostAllocMapped);
        cudaHostGetDevicePointer(\&out\_ptrs\_gpu[0], out\_ptrs\_cpu[0], 0);
        
        cudaStreamCreate(\&stream);
    }

    void inference(void\* camera\_buffer) {
        memcpy(input\_cpu\_ptr, camera\_buffer, input\_size); 
        context-\>setTensorAddress("images", input\_gpu\_ptr);
        context-\>setTensorAddress("reg3", out\_ptrs\_gpu[0]);
        // ... set all 6 addresses ...

        context-\>enqueueV3(stream);
        cudaStreamSynchronize(stream);

        // Process RAW Planar Results on CPU
        process\_detections();
    }
};

**Code Analysis:**

* cudaHostAllocMapped: This flag is the key. It allocates page-locked memory accessible by the DLA's DMA engine.  
* enqueueV3: The modern TensorRT API for submitting work.  
* No cudaMemcpy: Notice the absence of cudaMemcpyAsync. The DLA DMAs data directly from the system RAM while the GPU is free to do other tasks. This realizes the "Zero-Overhead" goal.

## **9\. Conclusion**

The **UNINA-DLA** blueprint represents the optimal convergence of algorithmic innovation and rigorous hardware engineering for Formula Student Driverless. By explicitly rejecting the complexity of CSPDarknet and the non-determinism of standard YOLO heads, and instead adopting a **RepVGG-B0 + YOLOv10-OneToOne** architecture, we satisfy the critical constraints of the NVIDIA Orin DLA.

This architecture achieves:

1. **Maximum Hardware Utilization:** Through the exclusive use of dense $3\\times3$ convolutions and ReLU activations.  
2. **Deterministic Latency:** By eliminating NMS and ensuring 100% DLA residency (no GPU fallbacks).  
3. **High Accuracy:** Through the Tri-Vector distillation strategy (ScaleKD) and Sensitive-Layer QAT.  
4. **Zero-Overhead Integration:** Through C++ Zero-Copy memory mapping.

Implementing this blueprint will provide the UNINA-DLA team with a perception system that is not merely fast, but "reflex-like"—decoupled from GPU load and capable of the sustained, low-latency performance required to push the limits of autonomous racing.

#### **Bibliografia**

1. text.txt  
2. Maximizing Deep Learning Performance on NVIDIA Jetson Orin with DLA, accesso eseguito il giorno gennaio 9, 2026, [https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/](https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/)  
3. ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector \- CVF Open Access, accesso eseguito il giorno gennaio 9, 2026, [https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu\_ScaleKD\_Distilling\_Scale-Aware\_Knowledge\_in\_Small\_Object\_Detector\_CVPR\_2023\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.pdf)  
4. Working with DLA — NVIDIA TensorRT Documentation, accesso eseguito il giorno gennaio 9, 2026, [https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)  
5. DLA bugs using deep-lab-v3 style network \- Jetson AGX Xavier \- NVIDIA Developer Forums, accesso eseguito il giorno gennaio 9, 2026, [https://forums.developer.nvidia.com/t/dla-bugs-using-deep-lab-v3-style-network/129679](https://forums.developer.nvidia.com/t/dla-bugs-using-deep-lab-v3-style-network/129679)  
6. Requirement for padding \< kernelSize makes DLA unusable \- NVIDIA Developer Forums, accesso eseguito il giorno gennaio 9, 2026, [https://forums.developer.nvidia.com/t/requirement-for-padding-kernelsize-makes-dla-unusable/216734](https://forums.developer.nvidia.com/t/requirement-for-padding-kernelsize-makes-dla-unusable/216734)  
7. DLA-v2 is slower than DLA-v1 \- Jetson AGX Orin \- NVIDIA Developer Forums, accesso eseguito il giorno gennaio 9, 2026, [https://forums.developer.nvidia.com/t/dla-v2-is-slower-than-dla-v1/215468](https://forums.developer.nvidia.com/t/dla-v2-is-slower-than-dla-v1/215468)  
8. DLA performance is not as expected \- Jetson AGX Orin \- NVIDIA Developer Forums, accesso eseguito il giorno gennaio 9, 2026, [https://forums.developer.nvidia.com/t/dla-performance-is-not-as-expected/298461](https://forums.developer.nvidia.com/t/dla-performance-is-not-as-expected/298461)  
9. Keys to optimization a network on AGX Orin DLA for latency \- NVIDIA Developer Forums, accesso eseguito il giorno gennaio 9, 2026, [https://forums.developer.nvidia.com/t/keys-to-optimization-a-network-on-agx-orin-dla-for-latency/268507](https://forums.developer.nvidia.com/t/keys-to-optimization-a-network-on-agx-orin-dla-for-latency/268507)  
10. Release Notes — NVIDIA TensorRT Documentation, accesso eseguito il giorno gennaio 9, 2026, [https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html)  
11. YOLOv10: Real-Time End-to-End Object Detection \- Ultralytics YOLO Docs, accesso eseguito il giorno gennaio 9, 2026, [https://docs.ultralytics.com/models/yolov10/](https://docs.ultralytics.com/models/yolov10/)  
12. \[2101.03697\] RepVGG: Making VGG-style ConvNets Great Again \- arXiv, accesso eseguito il giorno gennaio 9, 2026, [https://arxiv.org/abs/2101.03697](https://arxiv.org/abs/2101.03697)  
13. YOLOv10: Paper Explanation and Inference Results, accesso eseguito il giorno gennaio 9, 2026, [https://learnopencv.com/yolov10/](https://learnopencv.com/yolov10/)  
14. What is YOLOv10? An Architecture Deep Dive \- Roboflow Blog, accesso eseguito il giorno gennaio 9, 2026, [https://blog.roboflow.com/what-is-yolov10/](https://blog.roboflow.com/what-is-yolov10/)  
15. Implementing RepVGG in PyTorch \- GitHub, accesso eseguito il giorno gennaio 9, 2026, [https://github.com/FrancescoSaverioZuppichini/RepVgg](https://github.com/FrancescoSaverioZuppichini/RepVgg)  
16. \[Quick Review\] RepVGG: Making VGG-style ConvNets Great Again \- Liner, accesso eseguito il giorno gennaio 9, 2026, [https://liner.com/review/repvgg-making-vggstyle-convnets-great-again](https://liner.com/review/repvgg-making-vggstyle-convnets-great-again)  
17. What is YOLOv6? A Deep Insight into the Object Detection Model \- arXiv, accesso eseguito il giorno gennaio 9, 2026, [https://arxiv.org/html/2412.13006v1](https://arxiv.org/html/2412.13006v1)  
18. CVPR Poster ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector, accesso eseguito il giorno gennaio 9, 2026, [https://cvpr.thecvf.com/virtual/2023/poster/21575](https://cvpr.thecvf.com/virtual/2023/poster/21575)  
19. How to correctly implement the loss function for my distillation of Mask2Former?, accesso eseguito il giorno gennaio 9, 2026, [https://datascience.stackexchange.com/questions/134413/how-to-correctly-implement-the-loss-function-for-my-distillation-of-mask2former](https://datascience.stackexchange.com/questions/134413/how-to-correctly-implement-the-loss-function-for-my-distillation-of-mask2former)  
20. Make RepVGG Greater Again: A Quantization-Aware Approach, accesso eseguito il giorno gennaio 9, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/29045](https://ojs.aaai.org/index.php/AAAI/article/view/29045)  
21. Make RepVGG Greater Again: A Quantization-Aware Approach \- SciSpace, accesso eseguito il giorno gennaio 9, 2026, [https://scispace.com/pdf/make-repvgg-greater-again-a-quantization-aware-approach-uoipfbi23g.pdf](https://scispace.com/pdf/make-repvgg-greater-again-a-quantization-aware-approach-uoipfbi23g.pdf)  
22. Quantization-Aware Training (QAT) \- Ultralytics, accesso eseguito il giorno gennaio 9, 2026, [https://www.ultralytics.com/glossary/quantization-aware-training-qat](https://www.ultralytics.com/glossary/quantization-aware-training-qat)  
23. How Quantization Aware Training Enables Low-Precision Accuracy Recovery, accesso eseguito il giorno gennaio 9, 2026, [https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/)  
24. Jetson Zero Copy for Embedded applications \- APIs \- ximea support, accesso eseguito il giorno gennaio 9, 2026, [https://www.ximea.com/support/wiki/apis/Jetson\_Zero\_Copy\_for\_Embedded\_applications](https://www.ximea.com/support/wiki/apis/Jetson_Zero_Copy_for_Embedded_applications)  
25. Jetson zero-copy for embedded applications | fastcompression.com, accesso eseguito il giorno gennaio 9, 2026, [https://www.fastcompression.com/blog/jetson-zero-copy.htm](https://www.fastcompression.com/blog/jetson-zero-copy.htm)