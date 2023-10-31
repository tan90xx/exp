#!/bin/bash

# 检查参数是否正确
if [ $# -ne 1 ]; then
    echo "用法: $0 <音频文件目录>"
    exit 1
fi

# 设置音频文件目录
folder_path="$1"

# 定义文件夹路径
# folder_path="/root/autodl-tmp/sig1/blind_data"
# folder_path="/root/autodl-tmp/sig1/dev_data"

# 创建CSV文件并写入标题行
base_folder_path=$(basename "$folder_path")
output_csv="${base_folder_path}_audio_stats.csv"
echo "File Name,Rough Frequency" > "$output_csv"

# 遍历文件夹下的所有.wav文件，只统计前10条
# for input_file in $(ls "$folder_path"/*.wav | head -n 10)
for input_file in $(ls "$folder_path"/*.wav)
do

# 使用SoX的stat功能获取Rough frequency
frequency=$(sox "$input_file" -n stat 2>&1 | awk '/^Rough   frequency:/ {print $3}')

# 检查是否成功获取到频率
if [ -n "$frequency" ]; then
    # 获取文件名部分（不包含路径和扩展名）
    base_filename=$(basename "$input_file")

    # 创建新的文件名，将Rough frequency添加到文件名之前并加上"Hz"后缀
    # new_filename="${frequency}Hz_${base_filename}.wav"

    # 获取文件的目录路径
    # file_directory=$(dirname "$input_file")

    # 构建新的完整文件路径
    # new_file_path="${file_directory}/${new_filename}"

    # 重命名音频文件
    # mv "$input_file" "$new_file_path"

    #echo "已将文件重命名为 $new_file_path"
    
    # 写入文件名和Rough frequency到CSV文件
    echo "$base_filename,$frequency" >> "$output_csv"
    
else
    echo "无法获取频率信息。"
fi

done

