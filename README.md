# Fine-tune PaddleOCR ภาษาไทย (th_PP-OCRv5_mobile_rec) แบบใช้งานจริง: จากศูนย์จนเทรนสำเร็จด้วย GPU

บทความนี้สรุปขั้นตอนที่ผมใช้ในการ **fine-tune PaddleOCR สำหรับงาน Text Recognition (อ่านข้อความจากภาพที่ crop มาแล้ว)** โดยใช้โมเดลภาษาไทย **PP-OCRv5 mobile rec** พร้อมเล่าปัญหาที่เจอระหว่างทางและวิธีแก้ให้ผ่านทั้งหมด

> เป้าหมาย: เทรนโมเดล Recognizer (REC) ให้เข้ากับโดเมนเรา โดยใช้ pretrained `.pdparams` แล้ว eval / export ใช้งานได้จริง

---

## สรุปหัวใจสำคัญ

- โครงสร้างข้อมูลต้องถูก: `train_data/rec/train|val + rec_gt_train.txt|rec_gt_val.txt`
- label ต้องคั่นด้วย **TAB (`\t`)**
- ใช้ pretrained `.pdparams` แล้วชี้ใน config
- ถ้าจะเทรน/ประเมินให้ “ไม่พัง” แนะนำ **ใช้ original dict ของโมเดล** ทั้ง train และ eval  
  (หลีกเลี่ยง `character_dict_path` แบบ custom ที่ทำให้ head shape mismatch)
- แก้ปัญหา dependency:
  - `libGL.so.1` → ใช้ `opencv-python-headless`
  - cuDNN / CUDA (driver แสดง CUDA 13.0) → ติดตั้ง PaddlePaddle GPU wheel ใหม่ให้ถูกชุด และเตรียม cuDNN ให้พร้อม
  - การใช้ validation ในการ eval ต้องเปลี่ยน config batch ให้เป็น 1 
- train ด้วย CPU 1.30 นาที ต่อ epoc และ T4 GPU 5 วิ ต่อ epoc

---

## 1) เตรียม Repository

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
```

## 2) กำหนด Folder Structure สำหรับ Training (Recognition)
```
train_data/rec/
  train/                  # รูปข้อความที่ crop มาแล้ว (1 คำ/1 บรรทัด ต่อ 1 รูป)
  val/
  rec_gt_train.txt         # label train
  rec_gt_val.txt           # label val
```
**รูปแบบไฟล์ label (สำคัญมาก)** 
ไฟล์ rec_gt_train.txt / rec_gt_val.txt ต้องเป็น:
```
train/xxx.jpg<TAB>ข้อความจริง
val/yyy.jpg<TAB>ข้อความจริง
```
ตัวอย่าง
```
train/img_000001.jpg	สวัสดีครับ
train/img_000002.jpg	TOTAL ฿ 1,234.50
val/img_000010.jpg	วันที่ 17/12/2025
```

## 3) สคริปต์ Python สำหรับ Generate Synthetic Dataset (ไทย + อังกฤษ + ตัวเลข + สัญลักษณ์)

แนะนำให้เก็บฟอนต์ไว้ในโฟลเดอร์เดียวกับ notebook/script:
```
Generate_Dataset/
  generate.ipynb
  fonts/
    NotoSansThai-*.ttf
    THSarabun*.ttf
    ...
```

## 4) ดาวน์โหลดและเตรียม Pretrained .pdparams
```
mkdir -p pretrain_models

# 1) wget แบบ follow redirect + retry + timeout
wget -O pretrain_models/th_PP-OCRv5_mobile_rec_pretrained.pdparams \
  --max-redirect=20 --timeout=60 --tries=20 --retry-connrefused \
  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/th_PP-OCRv5_mobile_rec_pretrained.pdparams"
```
ถ้ายัง fail ให้ลอง curl (มักผ่านกว่า):
```
curl -L --retry 20 --retry-delay 2 -o pretrain_models/th_PP-OCRv5_mobile_rec_pretrained.pdparams \
  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/th_PP-OCRv5_mobile_rec_pretrained.pdparams"
```
นี่คือโครงสร้าง folder 
```
pretrain_models/
  th_PP-OCRv5_mobile_rec_pretrained.pdparams
```
จากนั้นใน config ให้ชี้:
```
Global:
  pretrained_model: ./pretrain_models/th_PP-OCRv5_mobile_rec_pretrained.pdparams
```

## 5) ปัญหา dependency ที่เจอ และวิธีแก้
### 5.1 ปัญหา ImportError: libGL.so.1 ... (cv2 / OpenCV)

อาการ:

เทรนเริ่มไม่ได้ เพราะ albumentations -> cv2 แล้วระบบหา libGL.so.1 ไม่เจอ

วิธีแก้ที่ง่ายสุด เปลี่ยนไปใช้ OpenCV แบบ headless (แนะนำ):
```bash
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install -U opencv-python-headless
```

### 5.2 ปัญหา CUDA/Driver แสดง CUDA 13.0 แล้ว Paddle หา cuDNN ไม่เจอ

อาการ: RuntimeError: Cannot load cudnn shared library ... cudnnGetVersion

สาเหตุ: driver รองรับ CUDA ใหม่มาก แต่ environment ยังไม่ได้ติดตั้ง PaddlePaddle GPU wheel / cuDNN runtime ที่เข้ากัน

แนวทางแก้: ติดตั้ง PaddlePaddle GPU ใหม่ให้ถูกชุด (เช่น cu126 line) และ ติดตั้ง cuDNN runtime (ผ่าน conda) และตั้ง LD_LIBRARY_PATH
```bash
pip uninstall -y paddlepaddle paddlepaddle-gpu
pip install -U paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

conda install -y -c nvidia -c conda-forge cudnn=9.5 cuda-cudart=12.6
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```
ตรวจสอบ:
```
python -c "import paddle; print(paddle.__version__); print('cuda:', paddle.is_compiled_with_cuda())"
python -c "import paddle; paddle.utils.run_check()"
```


## 6) ทำไมต้อง “ใช้ original dict” ไม่ใช้ custom dict

บทเรียนสำคัญที่สุดจากรอบนี้:
```
การเปลี่ยน character_dict_path เป็น dict ใหม่ ทำให้ head ของโมเดล (CTC/GTC) ขนาดไม่ตรงกับ pretrained/checkpoint
ส่งผลให้โหลด weight head ไม่เข้า หรือโหลดแล้ว decode ผิด mapping → eval ออกมา “ห่วยผิดธรรมชาติ”

สัญญาณเตือนที่เคยเจอ:

shape ... not matched ... [111] vs [104]

ผลตามมา:

acc: 0.0

norm_edit_dis ต่ำมาก (ใกล้ 0)

แนวทางที่ถูกต้อง:

ถ้าต้องการได้ประโยชน์จาก pretrained มากที่สุด และกันปัญหา mismatch:

ใช้ original dict ของโมเดล ทั้งตอน train และ eval

หากจำเป็นต้อง custom dict จริง ๆ:

ต้องใช้ dict เดียวกัน 100% ตั้งแต่เริ่มจนจบ (ห้าม regenerate/overwrite)

และยอมรับว่า head จะต้องเรียนรู้ใหม่บางส่วน
```
