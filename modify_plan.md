整体策略

DOVE 目前直接从 diffusers 加载 CogVideoXTransformer3DModel（finetune/models/dove/lora_one_s1_trainer.py:33-48, lora_one_s2_trainer.py:37-53）。要加 Token Merge + Routing，核心是改写这部分模型装载逻辑，而不是 trainer/dataset。
你给出的 Router/ToMe 代码建立在 SAT 的 BaseModel/Mixin 体系上，与 diffusers 的实现方式不同，需要把关键组件（合并函数、Router、路由状态机）抽出来，嵌入 CogVideoX 官方 transformer 的 forward 流程。
建议的改造步骤

整理工具函数

把 bipartite_soft_matching_randframe、bipartite_soft_matching_rand2d、Router、RouterMixin 等无 SAT 依赖的部分抽成一个新模块（例如 finetune/models/dove/token_merge.py）。
清理或替换 SAT 专属依赖（BaseModel、instantiate_from_config、ColumnParallelLinear 等），只保留 ToMe merge、路由状态和 RestoreAdapter 这类通用逻辑。
自定义 Transformer 子类

复制一份 diffusers 源码里的 CogVideoXTransformer3DModel（位于你安装的 diffusers 包中 diffusers/models/transformers/cogvideox_transformer3d.py），放到仓库（例如 finetune/models/dove/cogvideox_transformer3d_tokenmerge.py）。
在其 forward 循环访问每一层时，引入 Router 逻辑：
在设定的 start_layer 处调用 router.tome_merge_and_route 对图像 token 做 merge；
中间层带着缩减后的 token 继续计算；
在 end_layer 处使用 router.end_route 恢复，并通过 RestoreAdapter 细调合并后的 token。
若需要额外元信息（例如 pos_index_image），在 forward 内维护即可；无需像 SAT mixin 那样依赖外部 feature_tap。
配置化开关

在 finetune/schemas/args.py 中新增参数（如 enable_token_merge、routes、merge_ratio、merge_start_layer、merge_end_layer 等），默认关闭。
在 trainer 加载模型前读取这些参数，选择实例化原始 CogVideoXTransformer3DModel 还是新的 Token Merge 版本。
Trainer 调整

finetune/models/dove/lora_one_s1_trainer.py:30-58 和 lora_one_s2_trainer.py:34-62 改为根据新参数加载自定义 transformer；其它流程（dataset、loss）不变。
如果需要给 Router 传入额外的高度/宽度/帧数信息，可从 self.state.transformer_config 拿到 patch 尺寸、采样分辨率，传给 Router 初始化。
推理与权重导出

确保 finetune/scripts/prepare_sft_ckpt.py 在导出模型时也能复制新的 transformer 结构。必要时在 prepare_ckpt_structure 内对 transformer 子目录做相应调整。
推理脚本 inference_script.py 载入模型时，引入与训练相同的配置开关，保证一致性。
验证与回归测试

先在单 GPU 上用小批次跑一个短程（几十 step）验证前传是否稳定（注意 patch_size_t 及帧数补帧逻辑）。
对比是否能恢复原本 1 步扩散的输出；再尝试不同 merge ratio/路由配置，确认 Router 状态机能正确收敛。
照这个路线改完，就能在 DOVE 原始训练框架上插入你提供的 Token Merge + Route 结构，同时保持与现有两阶段 SFT 训练兼容。