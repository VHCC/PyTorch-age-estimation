import argparse
import better_exceptions
from pathlib import Path
from contextlib import contextmanager
import urllib.request
import numpy as np
import cv2
import dlib
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import timeit
from model import get_model
from defaults import _C as cfg


from flask import Flask, request, render_template
from config import DevConfig
from datetime import datetime
import os

# 初始化 Flask 類別成為 instance
app = Flask(__name__)
UPLOAD_FOLDER = 'img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
app.config.from_object(DevConfig)

@app.route('/api/getHello', methods=['GET'])
def home():
    return "Hello IChen 201910"

@app.route("/api/postFile", methods=['POST'])
def submit():
    if request.method == 'POST':
        file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        f = request.files['file']
        f.save(os.path.join(file_dir, f.filename))

        main_fd_flask(os.path.join(file_dir, f.filename), f.filename)
        # print(os.path.join(file_dir, f.filename))
        os.remove(os.path.join(file_dir, f.filename))
        return 'upload done, ' + request.values['name']

def get_args():
    parser = argparse.ArgumentParser(description="*Age estimation demo*",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help="Model weight to be tested")
    parser.add_argument("-m", "--margin", type=float, default=0.2,
                        help="Margin around detected face for age-gender estimation")
    parser.add_argument("-i", "--img_dir", type=str, default="img/",
                        help="Target image directory; if set, images in image_dir are used instead of webcam")
    parser.add_argument("-o", "--output_dir", type=str, default="img_result/",
                        help="Output directory to which resulting images will be stored if set")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img, None


def yield_images_from_dir(img_dir):
    img_dir = Path(img_dir)

    for img_path in img_dir.glob("*.*"):
        print("=> img_path= ", str(img_path))
        print("=> img_path.name= ", img_path.name)
        img = cv2.imread(str(img_path), 1)
        print("=> img= ", img)
        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r))), img_path.name

def yield_images_from_dir_flask(img_path, fileName):
    # print("=> img_path= ", str(img_path))
    img = cv2.imread(str(img_path), 1)
    # print("=> img= ", img)
    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        yield cv2.resize(img, (int(w * r), int(h * r))), fileName

def main_fd():
    start = timeit.default_timer()
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    if args.output_dir is not None:
        if args.img_dir is None:
            raise ValueError("=> --img_dir argument is required if --output_dir is used")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    print("=check= torch.cuda.is_available, '{}'".format(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if resume_path is None:
        resume_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")

        if not resume_path.is_file():
            print(f"=> model path is not set; start downloading trained model to {resume_path}")
            url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
            urllib.request.urlretrieve(url, str(resume_path))
            print("=> download finished")

    if Path(resume_path).is_file():
        # print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    print("=> device '{}'".format(device))
    if device == "cuda":
        cudnn.benchmark = True

    model.eval()
    margin = args.margin
    img_dir = args.img_dir
    detector = dlib.get_frontal_face_detector()
    img_size = cfg.MODEL.IMG_SIZE
    image_generator = yield_images_from_dir(img_dir) if img_dir else yield_images()

    with torch.no_grad():
        for img, name in image_generator:
            # print(name)
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            # print((len(detected), img_size, img_size, 3))
            faces = np.empty((len(detected), img_size, img_size, 3))
            # print("faces= " , len(detected))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1)

                # draw results
                for i, d in enumerate(detected):
                    age_label = "{}".format(int(predicted_ages[i]))
                    # print(d, age_label)
                    draw_label(img, (d.left(), d.top()), age_label)

            if args.output_dir is not None:
                # newfilename = datetime.today().strftime('%Y%m%d_%H%M%S') + "_" + f.filename
                # print(name)
                output_path = output_dir.joinpath(datetime.today().strftime('%Y%m%d_%H%M%S') + "_" + name)
                # print(str(output_path))
                cv2.imwrite(str(output_path), img)
            else:
                cv2.imshow("result", img)
                key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)

                if key == 27:  # ESC
                    break
            stop = timeit.default_timer()
            print('exc Time: ', stop - start , " s")
            # print(datetime.today().strftime('%Y%m%d_%H%M%S'))


def main_fd_flask(file_path, fileName):
    start = timeit.default_timer()
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # if args.output_dir is not None:
    #     if args.img_dir is None:
    #         raise ValueError("=> --img_dir argument is required if --output_dir is used")
    #
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # # create model
    # print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    # model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    # print("=check= torch.cuda.is_available, '{}'".format(torch.cuda.is_available()))
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    # if resume_path is None:
    #     resume_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")
    #
    #     if not resume_path.is_file():
    #         print(f"=> model path is not set; start downloading trained model to {resume_path}")
    #         url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
    #         urllib.request.urlretrieve(url, str(resume_path))
    #         print("=> download finished")

    # if Path(resume_path).is_file():
        # print("=> loading checkpoint '{}'".format(resume_path))
        # checkpoint = torch.load(resume_path, map_location="cpu")
        # model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}'".format(resume_path))
    # else:
    #     raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    # print("=> device '{}'".format(device))
    # if device == "cuda":
    #     cudnn.benchmark = True

    # model.eval()
    margin = args.margin
    # img_dir = args.img_dir
    # detector = dlib.get_frontal_face_detector()
    img_size = cfg.MODEL.IMG_SIZE
    image_generator = yield_images_from_dir_flask(file_path, fileName)

    with torch.no_grad():
        for img, name in image_generator:
            # print(name)
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            # print((len(detected), img_size, img_size, 3))
            faces = np.empty((len(detected), img_size, img_size, 3))
            # print("faces= " , len(detected))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1)

                # draw results
                for i, d in enumerate(detected):
                    age_label = "{}".format(int(predicted_ages[i]))
                    # print(d, age_label)
                    draw_label(img, (d.left(), d.top()), age_label)

            if args.output_dir is not None:
                # newfilename = datetime.today().strftime('%Y%m%d_%H%M%S') + "_" + f.filename
                # print(name)
                output_path = output_dir.joinpath(datetime.today().strftime('%Y%m%d_%H%M%S') + "_" + name)
                # print(str(output_path))
                cv2.imwrite(str(output_path), img)
            # else:
            #     cv2.imshow("result", img)
            #     key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)
            #
            #     if key == 27:  # ESC
            #         break
            stop = timeit.default_timer()
            print('exc Time: ', stop - start , " s, fileName= ", fileName)
            # print(datetime.today().strftime('%Y%m%d_%H%M%S'))

# if __name__ == '__main__':
#     start = timeit.default_timer()
#     main_fd()

# 判斷自己執行非被當做引入的模組，因為 __name__ 這變數若被當做模組引入使用就不會是 __main__
if __name__ == '__main__':
    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    print("=check= torch.cuda.is_available, '{}'".format(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    resume_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")
    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(resume_path))

    print("=> device '{}'".format(device))
    if device == "cuda":
        cudnn.benchmark = True

    model.eval()

    detector = dlib.get_frontal_face_detector()

    app.run(host='localhost', port=8000, debug=True)
