from PIL import Image

def high_res(low_res_png, high_res_png, width, height):
    img = Image.open(low_res_png)
    high_res = img.resize((width, height), Image.BICUBIC)
    high_res.save(high_res_png)

if __name__ == "__main__":
    low_res_img = 'final_V_heatmap.png'
    high_res_img = 'final_V_heatmap_high.png'
    high_res(low_res_img, high_res_img, 200, 200)