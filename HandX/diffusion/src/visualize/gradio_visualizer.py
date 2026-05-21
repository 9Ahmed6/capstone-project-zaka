import gradio as gr
import rerun as rr
import time
import numpy as np

from .rerun_visualizer import RerunVisualizer

RERUN_VIEWER_URL = "https://app.rerun.io/version/0.25.1/index.html?url="
generated_motion = np.zeros((10, 60, 2, 21, 3))

def generate_motion(left_hand_desc, right_hand_desc, relation_desc):
    print("正在为以下描述生成动作 (Generating motion for the following descriptions):")
    print(f"  左手 (Left Hand): {left_hand_desc}")
    print(f"  右手 (Right Hand): {right_hand_desc}")
    print(f"  双手关系 (Relation): {relation_desc}")

    global generated_motion
    generated_motion = np.random.randn(10, 60, 2, 21, 3) * 0.1

    visualizer.visualize_bihandmotion(motion=generated_motion[0])

    return (
        gr.Radio(choices=[f"结果 {i+1}" for i in range(10)], value="结果 1", label="选择一个结果 (Select a Result)", visible=True),
        gr.Group(visible=True)
    )

def update_viewer(selected_result):
    """
    根据选择的结果按钮, 更新 iframe 的 src。

    Updates the iframe src based on the selected result button.
    """
    if not selected_result:
        return ""

    result_index = int(selected_result.split(" ")[1]) - 1

    rr.reset_time()
    visualizer.visualize_bihandmotion(motion=generated_motion[result_index])

# --- Gradio UI 定义 (Gradio UI Definition) ---
with gr.Blocks(theme=gr.themes.Soft(), title="Text to Two-Hands Motion") as demo:
    gr.Markdown(
        """
        # 文本到双手动作生成演示 (Text to Two-Hands Motion Synthesis Demo)
        请输入对左手、右手及其关系的描述, 以生成三维手部动作。
        Enter descriptions for the left hand, right hand, and their relationship to generate 3D hand motions.
        """
    )

    with gr.Row():
        left_hand_input = gr.Textbox(label="左手描述 (Left Hand Description)", placeholder="例如: 握拳 (e.g., making a fist)")
        right_hand_input = gr.Textbox(label="右手描述 (Right Hand Description)", placeholder="例如: 挥手 (e.g., waving)")
        relation_input = gr.Textbox(label="双手关系 (Two Hands Relation)", placeholder="例如: 左手在右手上方 (e.g., left hand is above the right hand)")

    generate_btn = gr.Button("生成动作 (Generate Motions)", variant="primary")

    with gr.Group(visible=False) as output_group:
        result_selector = gr.Radio(
            choices=[],
            label="选择一个结果 (Select a Result)",
            info="模型会生成10个不同的结果, 请选择一个查看。(The model generates 10 different results. Choose one to view it below.)"
        )

    # --- 事件监听器 (Event Listeners) ---
    generate_btn.click(
        fn=generate_motion,
        inputs=[left_hand_input, right_hand_input, relation_input],
        outputs=[result_selector, output_group]
    )

    result_selector.change(
        fn=update_viewer,
        inputs=result_selector,
    )

if __name__ == "__main__":
    rr.init("Visualization Hands Motion")
    server_uri = rr.serve_grpc()
    rr.serve_web_viewer(open_browser=True, connect_to=server_uri, web_port=19090)
    RERUN_VIEWER_URL += server_uri

    visualizer = RerunVisualizer()
    # 启动 Gradio 应用 (Launch the Gradio app)
    demo.launch()
