#!/bin/bash

# 读取 model_paths.txt 文件
MODEL_PATH_FILE="model_paths.txt"

# 检查文件是否存在
if [ ! -f "$MODEL_PATH_FILE" ]; then
  echo "文件 $MODEL_PATH_FILE 不存在！"
  exit 1
fi

# 遍历文件中的每一行（即每个模型路径）
while IFS= read -r model_path; do
    if [ -n "$model_path" ]; then
        echo "正在处理模型路径: $model_path"

        # 在此处运行你希望的命令，举个例子：
        # 假设你要运行一个 Python 脚本，并传入模型路径作为参数
        python TxGNN_add_fintune.py --model "$model_path"

        # 如果需要其他命令，可以根据需求修改
        # 例如清理 GPU 缓存：torch.cuda.empty_cache()
    fi
done < "$MODEL_PATH_FILE"

echo "所有模型路径已处理完成！"