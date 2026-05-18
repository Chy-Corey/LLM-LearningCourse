## FlashAttention

> [FlashAttention从V1到V3简明攻略 - MKY-门可意 - 博客园](https://www.cnblogs.com/menkeyi/p/18800377)
>
> [Flash Attention 为什么那么快？原理讲解_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1UT421k7rA/?vd_source=2cbd78fa4c29853f276b2711bac1e01c)
>
> [zhuanlan.zhihu.com/p/676655352](https://zhuanlan.zhihu.com/p/676655352)



### 一、引言：Transformer中的Attention计算与GPU内存瓶颈

Transformer模型在自然语言处理、计算机视觉等领域取得了巨大的成功。其核心机制之一就是自注意力（Self-Attention）。简单来说，Attention机制允许模型在处理序列数据时，为不同的位置赋予不同的权重，从而更好地捕捉序列内部的依赖关系。

让我们简单回顾一下Transformer中Attention的计算过程。对于给定的Query (Q)、Key (K)和Value (V)矩阵，Attention的计算公式如下：
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
其中，$d_k$是Key (或Query)的维度，用于缩放点积结果，防止softmax的输入过大。

#### 1. Attention计算的GPU运算流程与内存瓶颈

在实际的GPU运算中，这个简单的公式背后隐藏着复杂的内存操作。标准的Attention计算流程大致如下：

1. 将Query状态（query state）和Key状态（key state）从**高带宽内存（HBM）**加载到**静态随机存取存储器（SRAM）**进行矩阵乘法运算（QKT）。
2. 将计算结果S写回HBM。
3. GPU再次将S从HBM加载到SRAM计算Softmax。
4. 将Softmax结果P写回HBM。
5. 最后，将P和Value状态（value state）从HBM加载到SRAM进行矩阵乘法运算（PV）。
6. 最终的输出O写回HBM。

#### 2. GPU HBM与SRAM简介

![GPU Memory](.\image\02\GPU Memory.jpg)

- **HBM (High Bandwidth Memory):** 这是一种高容量、高带宽的内存，通常用于存储模型参数和中间计算结果。但其访问速度相对较慢。
- **SRAM (Static Random Access Memory):** 这是GPU核心上的高速缓存，容量较小但访问速度极快。GPU进行计算时，通常需要将数据从HBM加载到SRAM中。

#### 3. 内存瓶颈的挑战

SRAM的容量非常有限且成本高昂，无法一次性加载整个Query或Key状态。因此，GPU需要以小块的形式频繁地在HBM和SRAM之间进行数据传输。这种频繁的数据读写带来了以下问题：

- **速度慢：** HBM的访问速度远低于SRAM，大量的数据搬运显著降低了Attention的运算速度。
- **内存碎片化：** 频繁的小块数据加载和卸载可能导致内存碎片化，进一步影响性能。

为了解决这些内存瓶颈问题，研究人员提出了FlashAttention系列算法。接下来，我们将逐一介绍FlashAttention V1、V2和V3的优化方法。

### 二、FlashAttention V1的优化方法

Flash Attention全称为：Fast and Memory-Efficient Exact Attention with IO-Awareness。可以理解为：

1. Fast：加快训练速度
2. Memory-Efficient：高效运用显存，减少显存占用
3. Exact Attention：精确的Attention计算，不会降低Attention精度
4. IO-Awareness：通过IO感知实现，即提高IO效率

FlashAttention的目标就是减少HBM的读写（减少更慢速度的IO操作），实现效率、速度的提高。

FlashAttention V1通过一系列巧妙的优化策略，显著减少了HBM和SRAM之间的内存传输，从而提升了Attention的计算效率。

其核心优化方法包括以下三个方面：

#### 1. Kernel Fusion

**核心思想：** 将多个独立的GPU操作（Kernel）合并为一个定制的Kernel。

**具体做法与优势：**

在标准的Attention计算中，计算Query和Key的点积、缩放、Softmax以及与Value的乘积是作为多个独立的Kernel在GPU上执行的。每个Kernel的执行都需要启动和同步，并且涉及HBM和SRAM之间的数据传输。

FlashAttention V1将这些操作融合到一个定制的Kernel中。这样做的好处是：

- **减少Kernel启动和同步的开销：** 减少了GPU调度和同步的次数，降低了整体的执行时间。
- **减少中间结果的HBM读写：** 融合后的Kernel可以直接在SRAM中完成多个步骤的计算，避免了将中间结果频繁地写回HBM再读出的操作，从而大幅降低了内存带宽的需求。

简单来说，Kernel Fusion就像是将一条生产线上的多个环节整合到一个更高效的机器中，减少了物料在不同环节之间的搬运。

当然，全部中间过程在SRAM中计算，会导致内存溢出，这需要对矩阵进行分块计算。

#### 2. Backward Recomputation

**核心思想：** 在反向传播过程中，重新计算前向传播的部分结果，以减少存储需求。

**具体做法与优势：**

在训练深度学习模型时，反向传播需要计算梯度。对于标准的Attention机制，为了计算梯度，通常需要在前向传播过程中保存大量的中间结果（例如，QKT的结果和Softmax的结果）。这些中间结果会占用大量的HBM空间，尤其是在处理长序列时。

FlashAttention V1通过Backward Recomputation技术，在反向传播时重新计算这些中间结果，而不是直接从HBM中读取。虽然这会增加一些计算量，但显著减少了HBM的内存占用，使得模型可以处理更长的序列，或者在有限的GPU内存下使用更大的批次大小。

可以理解为，与其记住每一步的详细操作，不如在需要的时候重新推导一遍，从而节省了记忆空间。

#### 3. Softmax Tiling

**核心思想：** 将Softmax的计算分块（Tile）进行，以适应SRAM的容量限制。

**具体做法与优势：**

Softmax操作通常需要对整个序列的Attention分数进行归一化。在处理长序列时，Softmax的输入矩阵可能非常大，无法一次性加载到SRAM中进行计算。

FlashAttention V1采用Softmax Tiling的方法，将Attention分数矩阵分割成若干个小的块（Tile），然后逐个加载到SRAM中进行Softmax计算。在计算每个Tile的Softmax时，算法会利用之前计算的块的信息，以保证最终结果的正确性。

这种分块计算的方式使得即使在SRAM容量有限的情况下，也能够处理任意长度的序列的Softmax计算，避免了因内存不足而导致的问题。

#### 4. 总结

FlashAttention V1通过Kernel Fusion减少了计算和内存访问的开销，通过Backward Recomputation降低了反向传播的内存需求，并通过Softmax Tiling解决了长序列Softmax计算的内存限制问题，从而显著提升了Transformer模型的训练和推理效率。

好的，我们现在来看FlashAttention V2是如何在V1的基础上进一步优化的。

------

### 三、FlashAttention V2的优化方法

FlashAttention V2在FlashAttention V1的基础上，针对反向传播、因果掩码（Causal Mask）以及GPU的并行计算等方面进行了更深入的优化，进一步提升了性能。

#### 1. Backward优化

**核心思想：** 进一步优化反向传播的效率。

**具体做法与优势：**

FlashAttention V1已经通过Backward Recomputation减少了内存占用。FlashAttention V2在此基础上，对反向传播的计算流程进行了更细致的优化，例如更高效的梯度计算方式，减少了不必要的计算和内存访问，从而进一步提升了反向传播的速度。具体的优化细节可能涉及到更精细的Kernel设计和数据调度策略。

#### 2. Casual Mask优化

**核心思想：** 针对自回归模型中常用的因果掩码进行专门优化。

**背景知识：**

- **自回归模型 (Autoregressive Model):** 这类模型在生成序列时，通常按照时间步逐个生成，并且每个时间步的输出都依赖于之前时间步的输出。例如，在文本生成任务中，生成下一个词需要考虑前面已经生成的词。
- **因果掩码 (Causal Mask):** 在Transformer的自注意力机制中，为了保证自回归的特性，通常会使用因果掩码。这个掩码会阻止序列中的每个位置“看到”未来位置的信息。具体来说，对于序列中的第i个位置，只能attend到第i个位置以及之前的（1到i-1）位置。在Attention的计算中，这通常通过将未来位置的Attention分数设置为负无穷（或一个非常小的负数），使得在Softmax后这些位置的权重趋近于零来实现。

**具体做法与优势：**

FlashAttention V2针对这种因果掩码的特性进行了优化。在计算Attention时，它可以更有效地处理掩码带来的稀疏性，避免对被掩盖的位置进行不必要的计算和内存访问。这对于训练和推理自回归模型（如各种大型语言模型）来说至关重要，因为它们广泛使用因果掩码。

#### 3. Multi-Thread Block

**核心思想：** 更有效地利用GPU上的Streaming Multiprocessor (SM)。

**背景知识回顾：**

在FlashAttention V1的运算中，通常以一个Thread Block为单位进行计算。对于一个给定的批次大小（batch size）和注意力头数（attention head），总共会有`batch size * attention head`个Thread Block被分配到GPU的SM上运行。例如，Nvidia A100 GPU有108个SM。

**问题：长序列长度下的SM利用率不足**

当序列长度很长时，为了避免显存溢出，通常需要减小批次大小。同时，模型的注意力头数也可能相对较小。在这种情况下，总的Thread Block数量可能会远小于GPU上SM的数量，导致大量的SM处于空闲状态，无法充分利用GPU的并行计算能力。

**FlashAttention V2的优化：**

FlashAttention V2同样会根据序列长度进行划分，但它允许将多个逻辑上的Attention计算任务（原本可能对应多个Thread Block）分配给一个或一组SM来执行。这样即使在批次大小和注意力头数较小的情况下，也能更充分地利用GPU的SM资源，从而提高整体的计算吞吐量。

#### 4. Work Partitioning Between Warps

**核心思想：** 优化Thread Block内部Warps之间的工作分配和同步。

**背景知识回顾：**

在GPU的硬件层面，真正的运算单元是Warp，通常一个Warp包含32个线程（thread）。一个Thread Block通常包含多个Warps（例如4个或8个）。

**FlashAttention V1的潜在瓶颈：Warp同步**

在FlashAttention V1中，一个Thread Block内的不同Warps在计算Q和KV的矩阵乘法时，每个Warp需要将自己的计算结果先存储到Thread Block的共享内存（shared memory）中，然后所有Warps必须同步（synchronize）以确保所有结果都已计算完毕，才能将每个Warp的输出进行汇总。这个同步的过程可能会成为性能瓶颈，因为Warps之间需要互相等待。

**FlashAttention V2的优化：避免Warp同步**

FlashAttention V2改变了计算顺序，使得一个Thread Block内的每个Warp可以直接对KV进行矩阵相乘，而无需先将结果写入共享内存再进行同步。这样，每个Warp只需要负责自己那部分数据的计算，不需要等待其他Warps完成，从而减少了同步开销，加速了前向传播的运算。

#### 5. Tuning Block Size

**核心思想：** 针对不同的硬件和模型配置，优化Block Size的选择。

**具体做法与优势：**

FlashAttention V2的研究人员通过实验发现，对于不同的GPU型号和模型配置（例如，注意力头的维度和GPU的共享内存大小），选择合适的Block Size对于性能至关重要。他们指出，通常情况下，将Block Size设置为{64, 128} × {64, 128}会有较好的效果。然而，最佳的Block Size仍然需要根据具体的硬件和模型进行调整。FlashAttention V2提供了一些指导原则和自动调优的机制，帮助用户找到最佳的Block Size配置。

**总结**

FlashAttention V2通过对反向传播、因果掩码、GPU资源利用以及工作分配等方面的细致优化，进一步提升了Transformer模型的性能，尤其在处理长序列和自回归模型方面表现更加出色。

### 四、FlashAttention V3的优化方法

FlashAttention V2在像A100这样的Ampere架构GPU上已经做了高度优化，但随着GPU技术的不断发展，Nvidia推出了更强大的Hopper架构GPU，例如H100。令人惊讶的是，FlashAttention V2在H100上的性能利用率仅为35%左右。为了充分利用新一代GPU的强大能力，FlashAttention V3应运而生，其可以将H100的GPU利用率提高到75%，速度比FlashAttention V2提升了1.5到2倍。

FlashAttention V3的优化主要集中在以下几个方面：

#### 1. Thread Block Cluster

**核心思想：** 引入更细粒度的线程调度和内存管理机制，以适应H100更强大的并行计算能力。

**背景知识回顾：**

在之前的A100 GPU上，线程通常被组织成三个层级：线程（thread）、线程块（thread block）和网格（Grid）。

**H100的新层级：Thread Block Cluster**

由于H100拥有更多的Streaming Multiprocessor (SM)和更强大的计算能力，仅仅使用三层级的线程组织方式已经无法满足更复杂和庞大的计算任务的需求。因此，H100引入了**Thread Block Cluster**这一新的层级。这个层级使得线程的调度和内存管理的粒度可以进一步细化。

**GEME (Global Memory):**

这里需要简单科普一下GEME。GEME可以理解为GPU的全局内存，它对应到硬件上的HBM。这是GPU上带宽最慢的内存区域，因此，为了加速运算，我们需要尽量减少从GEME中读取数据。

**Thread Block Cluster与硬件对应：**

- **Grid:** 可以对应到GEME（HBM）。
- **Thread Block Cluster:** 在硬件上可以对应到Graph Processing Cluster (GPC)。GPC提供了所谓的SM-to-SM Network，用于加速不同SM之间的数据传输。

**优势：更高效的跨SM数据传输**

在A100中，如果不同的Thread Block之间需要互相传递数据，通常需要通过HBM。但在H100中，借助Thread Block Cluster和SM-to-SM Network，可以直接在不同的SM之间进行更高效的数据传输。这些数据在物理上位于L2 Cache中，逻辑上被称为分布式共享内存（distributed shared memory, DSMEM）。

**小补充：** L1 Cache位于每个SM内部，而L0 Cache则位于每个Warp内部。

#### 2. Tensor Memory Accelerator (TMA)

**核心思想：** 实现HBM和SMEM之间的数据传输与计算的重叠，提高数据加载效率。

**具体做法与优势：**

Tensor Memory Accelerator (TMA)是H100上一个非常重要的全新特性。简单来说，TMA允许在HBM和共享内存（SMEM）之间同时进行数据传输和计算，实现了计算与通信的重叠（computation & communication overlap）。

在之前的A100上，如果SM需要从HBM加载数据到SMEM，SM必须先创建一个读取线程，然后才能将数据从HBM读入。而在H100上，数据读取的任务可以交给TMA来处理。这样一来，SM就可以释放更多的计算资源，专注于执行其自身的运算任务。这极大地提高了数据加载的效率，减少了GPU的等待时间。

#### 3. egister Dynamic Reallocation

**核心思想：** 在Warp Group（4个Warps）之间动态地重新分配寄存器，以提高寄存器的利用率。

**具体做法与优势：**

FlashAttention V3引入了寄存器动态重分配机制。这意味着在一个Warp Group（通常包含4个Warps）内部，寄存器可以根据需要动态地进行分配和回收。这使得模型在运行时可以拥有更多的寄存器资源（RMEM）可用，从而减少了对共享内存的依赖，并可能进一步提升计算性能。

#### 4. Warp-specialization

**核心思想：** 将Thread Block内的Warps划分为生产者（Producer）和消费者（Consumer）两种角色，实现数据加载和计算的流水线并行。

**数据流定义：**

我们可以将数据的传递过程看作生产者-消费者的模式：

- **生产者 (Producer):** 对应到TMA，负责从HBM加载数据到共享内存。
- **消费者 (Consumer):** 对应到Tensor Core，负责使用加载到共享内存的数据进行矩阵运算。

**具体做法与优势：**

FlashAttention V3将一个Thread Block内的Warps分成Producer Warp Group和Consumer Warp Group。

- **Producer Warp Group:** 使用TMA将数据从HBM加载到共享内存。
- **Consumer Warp Group:** 使用Tensor Core来计算这些数据。

在Consumer Warp Group中，存在两种不同的矩阵运算：

- **SS-GEMM:** 第一个操作数（例如Q）来自共享内存。
- **RS-GEMM:** 第一个操作数（例如P）来自寄存器。

由于需要先有Query (Q) 才能进行后续的计算，因此Q必须首先通过TMA从HBM加载到共享内存。对于Key (K) 和 Value (V)，FlashAttention V3采用了异步加载的方式。它会初始化一个s-stage的循环共享内存缓冲区（circular SMEM buffer）来记录加载到共享内存的KV数据。在进入生产者Warp Group的循环时，生产者会持续读取KV数据直到缓冲区满（经过s次），而无需等待消费者Warp Group是否已经处理完当前缓冲区的数据。当缓冲区满后，生产者才会等待消费者计算完Attention并释放当前stage的缓冲区，然后生产者再继续读取新的KV数据。

在计算相似度矩阵S时，数据源来自共享内存（SS-GEMM），而在计算最终输出O时，数据源来自寄存器（RS-GEMM）。值得一提的是，FlashAttention V3会利用寄存器动态重分配机制来增加可用的寄存器数量，这对于这种生产者-消费者模式下的异步操作非常重要。

#### 5. Pingpong scheduling

**核心思想：** 将矩阵运算和Softmax运算并行执行，进一步提高GPU的利用率。

**具体做法与优势：**

FlashAttention V3的作者对仅仅实现数据加载和计算的并行还不满足，他们进一步实现了矩阵运算和Softmax运算的同步并行。这主要是因为Softmax中的指数运算（exp）是由GPU的multi-function unit执行的，这意味着当Tensor Core进行矩阵运算时，multi-function unit可以同时进行Softmax的计算。

FlashAttention V3将Attention的计算过程主要分为三个步骤：

1. **GEMM0:** Query (Q) 和 Key (KT) 的矩阵乘法。
2. **Softmax:** 计算Attention权重P。
3. **GEMM1:** Attention权重P和Value (V) 的矩阵乘法。

通过Pingpong scheduling，FlashAttention V3在使用两个Warp Group时，可以强制Warp Group 2完成GEMM0后，Warp Group 1才能进行GEMM1。这样做的目的是为了让两个Warp Group交替执行不同的计算任务，使得在执行矩阵乘法的同时，另一个Warp Group可以执行Softmax运算，从而充分利用GPU的计算资源，避免了在计算Softmax时Tensor Core处于空闲状态的情况。

**总结**

FlashAttention V3通过引入Thread Block Cluster、Tensor Memory Accelerator、Register Dynamic Reallocation、Warp-specialization和Pingpong scheduling等一系列针对H100 GPU架构的创新优化，极大地提高了Transformer模型的性能，尤其是在处理大规模数据和运行在最新一代GPU上时，展现出惊人的效率提升。

### 五、总结：

从FlashAttention V1到V3，我们可以清晰地看到研究人员在不断探索如何更有效地利用GPU的硬件特性来加速Transformer模型的Attention计算。

- **FlashAttention V1** 通过Kernel Fusion、Backward Recomputation和Softmax Tiling等方法，首次显著地减少了HBM和SRAM之间的内存传输，为后续的优化奠定了基础。
- **FlashAttention V2** 在V1的基础上，针对反向传播、因果掩码以及GPU的并行计算等方面进行了更深入的优化，尤其在处理长序列和自回归模型上表现更佳。
- **FlashAttention V3** 则完全针对最新的H100 GPU架构进行了定制化的优化，通过引入Thread Block Cluster、TMA、寄存器动态重分配、Warp-specialization和Pingpong scheduling等先进技术，将GPU的利用率提升到了一个全新的高度，实现了惊人的性能提升。

通过不断地探索和优化，FlashAttention不仅提升了模型的训练和推理速度，也为我们展示了如何深入理解硬件特性并将其应用于算法设计，从而实现极致的性能优化。随着GPU技术的不断进步，我们有理由相信，未来还会涌现出更多像FlashAttention一样优秀的算法，进一步推动人工智能领域的发展。