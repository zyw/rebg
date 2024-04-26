# This is a sample Python script.
import io
import os

from PIL import ImageColor, Image
from rembg import remove, new_session
import gradio as gr

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

dir_path = os.getcwd()
# input_path = 'input.png'
# output_path = os.path.join(dir_path, 'output')


def open_file(input_path,
              model_name="u2net",
              alpha_matting=False,
              alpha_matting_foreground_threshold=270,
              alpha_matting_background_threshold=20,
              alpha_matting_erode_size=11,
              only_mask=False,
              invert_mask=False,
              post_process_mask=False,
              bgcolor=None
              ):
    # with open(input_path, 'rb') as i:
    #     temp_path = os.path.join(output_path, os.path.basename(input_path))
    #     with open(temp_path, 'wb') as o:
    #         input = i.read()
    #         output = remove(input)
    #         o.write(output)
    #
    #     return temp_path
    session = new_session(model_name)
    if invert_mask:
        output = remove(input_path, session=session,
                        alpha_matting=alpha_matting,
                        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                        alpha_matting_background_threshold=alpha_matting_background_threshold,
                        alpha_matting_erode_size=alpha_matting_erode_size,
                        only_mask=True,
                        post_process_mask=post_process_mask,
                        bgcolor=bgcolor)
        return Image.fromarray(255 - output)

    return remove(input_path, session=session,
                  alpha_matting=alpha_matting,
                  alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                  alpha_matting_background_threshold=alpha_matting_background_threshold,
                  alpha_matting_erode_size=alpha_matting_erode_size,
                  only_mask=only_mask,
                  post_process_mask=post_process_mask,
                  bgcolor=bgcolor)


# 设置U2NET_HOME环境变量
def set_model_env():
    os.environ['U2NET_HOME'] = os.path.join(dir_path, '.u2net')
    getenv = os.getenv('U2NET_HOME')
    print("当前U2NET_HOME路径为：", getenv)


# 使用gradio生成一个图片生成的UI界面
def generate_img(image_input,
                 images_input,
                 model_name,
                 alpha_matting,
                 foreground_threshold,
                 background_threshold,
                 erode_size,
                 mask_invert_mask,
                 post_process_mask,
                 bgcolor,
                 bg_opacity):
    print("model_name, ", model_name)
    print("alpha_matting, ", alpha_matting)
    print("mask_invert_mask, ", mask_invert_mask)
    print("post_process_mask, ", post_process_mask)
    if bgcolor is None:
        bgcolor = (0, 0, 0, 0)
    else:
        bgcolor = ImageColor.getcolor(bgcolor, "RGB") + (bg_opacity,)
    print("bgcolor, ", bgcolor)
    only_mask = False
    invert_mask = False
    if mask_invert_mask == 1:
        only_mask = True
    elif mask_invert_mask == 2:
        invert_mask = True
    image_output_gallery = []
    if images_input is not None:
        for image_path in images_input:
            image_output = open_file(image_path,
                                     model_name=model_name,
                                     alpha_matting=alpha_matting,
                                     alpha_matting_foreground_threshold=foreground_threshold,
                                     alpha_matting_background_threshold=background_threshold,
                                     alpha_matting_erode_size=erode_size,
                                     only_mask=only_mask,
                                     invert_mask=invert_mask,
                                     post_process_mask=post_process_mask,
                                     bgcolor=bgcolor)
            image_output_gallery.append(Image.open(io.BytesIO(image_output)))
        return image_output_gallery

    return [open_file(image_input,
                      model_name=model_name,
                      alpha_matting=alpha_matting,
                      alpha_matting_foreground_threshold=foreground_threshold,
                      alpha_matting_background_threshold=background_threshold,
                      alpha_matting_erode_size=erode_size,
                      only_mask=only_mask,
                      invert_mask=invert_mask,
                      post_process_mask=post_process_mask,
                      bgcolor=bgcolor)]
    # return image_input


def create_input_gui():
    # 创建一个UI界面，输入图片和抠图按钮
    with gr.Blocks(css=".img-files {height: 350px !important;}") as rembg_ui:
        with gr.Row():
            with gr.Tab(label="上传图片"):
                image_input = gr.Image(label="上传图片", show_label=False, sources=["upload"], elem_classes="img-files")
            with gr.Tab(label="选择多个图片"):
                images_input = gr.File(label="选择图片", show_label=False, type="binary", elem_classes="img-files",
                                       file_count="multiple", file_types=["image"])
            # image_output = gr.Image(format="png", label="抠图结果", width=400, height=400)
            image_output_gallery = gr.Gallery(format="png", label="抠图结果", elem_id="gallery", preview=True, columns=[3],
                                              rows=[1], object_fit="contain", height=400)
        with gr.Row():
            model_name = gr.Dropdown(label="选择模型",
                                     choices=["u2net",
                                              "u2netp",
                                              "u2net_human_seg",
                                              "u2net_cloth_seg",
                                              "silueta",
                                              "isnet-general-use",
                                              "isnet-anime"], value="u2net")
            alpha_matting = gr.Checkbox(label="alpha", info="是否使用Alpha遮罩的标志?")
            mask_invert_mask = gr.Radio([("非蒙版", 0), ("蒙版", 1), ("反转蒙版", 2)], value=0, label="返回蒙版",
                                        info="返回蒙版还是蒙版的反转?")
            post_process_mask = gr.Checkbox(label="蒙版后处理", info="是否对蒙版进行后处理?")
        with gr.Row():
            bg_opacity = gr.Slider(0, 255, value=0, step=1, label="背景透明度", info="background opacity")
            bgcolor = gr.ColorPicker(label="背景色", info="蒙版背景色，默认为黑色", scale=1)
        with gr.Row():
            foreground_threshold = gr.Slider(0, 255, value=240, step=1, label="前景阈值", info="foreground threshold")
            background_threshold = gr.Slider(0, 255, value=10, step=1, label="背景阈值", info="background threshold")
            erode_size = gr.Slider(0, 40, value=10, step=1, label="侵蚀大小", info="erode size")

        image_button = gr.Button("生 成", variant="primary")
        image_button.click(generate_img, inputs=[
            image_input,
            images_input,
            model_name,
            alpha_matting,
            foreground_threshold,
            background_threshold,
            erode_size,
            mask_invert_mask,
            post_process_mask,
            bgcolor,
            bg_opacity
        ], outputs=image_output_gallery)
        # img = gr.Interface(greet, create_input_gui(), "image", live=True, batch=True)
    rembg_ui.title = "基于Rembg的抠图工具"
    rembg_ui.launch(share=True, inbrowser=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    set_model_env()
    create_input_gui()
