# argument.py

import argparse
import os

image_base_dir = os.getenv("SWIG_IMAGE_DIR", "data/swig/images")
api_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
api_key = os.getenv("OPENAI_API_KEY", "")
model = os.getenv("AGENTHOI_MODEL", "gpt-4o-2024-11-20")
model_name = os.getenv("AGENTHOI_MODEL_NAME", "4o")
base_dir = os.getenv("SWIG_OUTPUT_DIR", f"swig_pipe/output/{model_name}")
concurrency = 8
max_tokens = 1500


def initial_HOI_Identification_arguments():
    """
    定义并解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="使用OpenAI API异步分析图片，支持定期保存和错误重处理。")
    
    # 文件与目录路径参数
    parser.add_argument("--image_base_dir", type=str, default=image_base_dir, 
                        help="图片存储的基础目录。")
    parser.add_argument("--annotation_file", type=str, default="data/swig/swig_test_ann_clarification.json", 
                        help="列出图片文件名的注解JSON文件路径。")
    parser.add_argument("--prompt_file", type=str, default="swig_pipe/prompt/Initial_HOI_Identification_prompt.txt", 
                        help="包含系统任务提示的文件路径。")

    # OpenAI API 相关参数
    parser.add_argument("--api_key", type=str, default=api_key,
                        help="OpenAI API密钥。默认为环境变量 OPENAI_API_KEY。")
    parser.add_argument("--api_base_url", type=str, default=api_base_url,
                        help="可选的OpenAI API基础URL。默认为环境变量 OPENAI_API_BASE_URL 或OpenAI官方地址。")
    parser.add_argument("--openai_model", type=str, default=model,
                        help="要使用的OpenAI模型 (例如 gpt-4o, gpt-4-turbo)。默认: gpt-4o。")
    parser.add_argument("--max_tokens_per_image", type=int, default=max_tokens, 
                        help="每张图片API响应的最大token数。")
    # 处理流程控制参数
    parser.add_argument("--concurrency", type=int, default=concurrency, 
                        help="同时发送给OpenAI API的并发请求数量 (默认: 10)。")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="每处理N张新图片后保存一次结果 (默认: 10)。")
    
    parser.add_argument("--output_file", type=str, default= os.path.join(base_dir, f"{model_name}_p1.json"),
                help="存储结果的输出JSON文件路径 (默认: image_analysis_results.json)。")
    
    args = parser.parse_args()
    return args

def structured_answer_arguments():
    """
    定义并解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="整理第一轮LLM结果")

    # 1. 定义 stage 参数
    parser.add_argument('--stage', type=int, default=1, help='选择要运行的阶段 (1 或 2)', choices=[1, 2])

    # 2. 定义 input_file 和 output_file，初始 default 为 None
    parser.add_argument("--input_file", type=str, default=None,
                        help="输入文件。如果未提供，将根据 'stage' 参数自动生成。")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件。如果未提供，将根据 'stage' 参数自动生成。")
    parser.add_argument("--failed_file", type=str, default="swig_pipe/output/p1/failed_p2.json",
                        help="包含系统任务提示的文件路径。")

    # 3. 解析所有参数
    args = parser.parse_args()

    # 4. 后处理：如果 input_file 或 output_file 为 None (即用户未指定)，
    #    则根据解析出来的 args.stage 来构建它们的默认值。
    if args.input_file is None:
        args.input_file = os.path.join(base_dir, f"{model_name}_p{args.stage}.json")
    if args.output_file is None:
        args.output_file = os.path.join(base_dir, f"{model_name}_p{args.stage}_ps1.json")
    return args


def filter_object_arguments():
    """
    定义并解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="整理第一轮LLM结果")
    
    parser.add_argument('--stage', type=int, default=1, help='选择要运行的阶段 (1 或 2)', choices=[1, 2])
    
    # 文件与目录路径参数
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    
    if args.input_file is None:
        args.input_file = os.path.join(base_dir, f"{model_name}_p{args.stage}_ps1.json")
    if args.output_file is None:
        args.output_file = os.path.join(base_dir, f"{model_name}_p{args.stage}_ps2.json")
    
    
    return args

def object_logits_update_arguments():
    parser = argparse.ArgumentParser(description="整理参数")
    
    # 文件与目录路径参数
    # 
    parser.add_argument("--refer_file", type=str, default=os.path.join(base_dir, f"{model_name}_p1_ps2.json"))
    parser.add_argument("--input_file", type=str, default=os.path.join(base_dir, f"{model_name}_p2_ps2.json"))
    parser.add_argument("--output_file", type=str, default=os.path.join(base_dir, f"{model_name}_p2_ps2.json"))
    
    args = parser.parse_args()
    
    return args
    
    
def HOI_Remining_arguments():
    parser = argparse.ArgumentParser(description="使用 OpenAI API 异步处理图片。")
    
    # 配置参数
    parser.add_argument('--input_file', type=str, default=os.path.join(base_dir, f"{model_name}_p1_ps2.json"))
    parser.add_argument('--output_file', type=str, default=os.path.join(base_dir, f"{model_name}_p2.json"))
    
    
    parser.add_argument('--prompt_file', type=str, default='swig_pipe/prompt/HOI_Remining_prompt.txt',
                        help='包含系统任务提示的文件路径。')
    parser.add_argument('--image_base_dir', type=str,
                        default=image_base_dir,
                        help='存储图片的基目录。')
    parser.add_argument('--api_key', type=str, default=api_key,
                        help='OpenAI API 密钥。也可以通过 OPENAI_API_KEY 环境变量设置。')
    parser.add_argument('--api_base_url', type=str, default=api_base_url,
                        help='OpenAI API 的基础 URL。')
    parser.add_argument('--model', type=str, default=model,
                        help='用于分析的 OpenAI 模型。')
    parser.add_argument('--max_tokens', type=int, default=max_tokens,
                        help='API 响应的最大 token 数。')
    parser.add_argument('--concurrency', type=int, default=concurrency,
                        help='并发 API 调用数。')
    parser.add_argument('--save_interval', type=int, default=10, # 稍微增大默认保存间隔
                        help='在本会话中处理这么多新图片后就保存结果到文件。')

    args = parser.parse_args()
    return args

def combine_same_HOI():
    parser = argparse.ArgumentParser(description="整理第一轮LLM结果")
    parser.add_argument("--input_file", type=str, default=os.path.join(base_dir, f"{model_name}_p2_ps2.json"))
    parser.add_argument("--output_file", type=str, default=os.path.join(base_dir, f"{model_name}_p2_ps2.json"))
    args = parser.parse_args()
    return args
    
def Action_Reassignment_arguments():
    parser = argparse.ArgumentParser(description="使用 OpenAI API 异步更新 HICO 数据集中的交互类别 (IC)。")
    
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join(base_dir, f"{model_name}_p2_ps2.json"))
    parser.add_argument('--output_main_file', type=str, 
                        default=os.path.join(base_dir, f"{model_name}_p3.json"))
    
    
    parser.add_argument('--output_ic_result_file', type=str, 
                        default=os.path.join(base_dir, f"{model_name}_icresult.json"))
    
    parser.add_argument('--prompt_file', type=str, default='swig_pipe/prompt/p3_prompt.txt',
                        help='包含系统任务提示的文件路径。')
    parser.add_argument('--image_base_dir', type=str,
                        default=image_base_dir, # 源脚本默认值
                        help='存储图片的基目录。')
    
    parser.add_argument('--api_key', type=str, default=api_key,
                        help='OpenAI API 密钥。')
    parser.add_argument('--api_base_url', type=str, default=api_base_url, # 源脚本默认值
                        help='OpenAI API 的基础 URL。')
    parser.add_argument('--model', type=str, default=model, # 源脚本默认值
                        help='用于分析的 OpenAI 模型。')
    parser.add_argument('--max_tokens', type=int, default=300, # 源脚本默认值
                        help='API 响应的最大 token 数。')
    parser.add_argument('--concurrency', type=int, default=concurrency,
                        help='并发 API 调用数。')
    parser.add_argument('--save_interval', type=int, default=10, # 源脚本默认是1，这里设为10
                        help='在本会话中处理这么多新图片后就保存结果到文件。')

    args = parser.parse_args()
    return args

def extract_ic_logits_arguments():
    """
    定义并解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="整理动作的logits结果")
    
    # 文件与目录路径参数
    parser.add_argument("--input_file", type=str, default=os.path.join(base_dir, f"{model_name}_p3.json"))
    parser.add_argument("--output_file", type=str, default=os.path.join(base_dir, f"{model_name}_p3_ps3.json"))
    parser.add_argument("--failed_file", type=str, default=os.path.join(base_dir, f"{model_name}_failed_p3.json"))

    args = parser.parse_args()
    return args

def Box_arguments():
    parser = argparse.ArgumentParser(description="GroundingDINO出框")
    parser.add_argument('--image_folder', type=str,
                        default=image_base_dir, # From first script
                        help='Path to image folder')
    parser.add_argument('--input_file', type=str,
                        default=os.path.join(base_dir, f"{model_name}_p3_ps3.json"))
    parser.add_argument('--output_file', type=str,
                        default=os.path.join(base_dir, f"{model_name}_box.json"))
    parser.add_argument('--num_workers', type=int, default=7, choices=range(1, 11),
                        help='Number of worker processes (GPUs) to use. Defaults to available GPU count.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, # From first script
                        help='Number of images to process before saving a checkpoint')

    args = parser.parse_args()
    return args
    # 配置参数

if __name__ == '__main__':
    # 这个部分只在直接运行 argument.py 时执行，可以用于测试参数解析
    print("测试参数解析 (如果直接运行此文件):")
    test_args = initial_HOI_Identification_arguments()
    print(f"图片基础目录: {test_args.image_base_dir}")
    print(f"注解文件: {test_args.annotation_file}")
    # ... 可以打印其他参数进行测试
