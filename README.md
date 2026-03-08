# So sánh mBERT và PhoBERT cho Bài toán Phân loại Cảm xúc Tiếng Việt

## Giới thiệu

Dự án này thực hiện **bài toán phân loại cảm xúc (Sentiment Analysis)** trên dữ liệu phản hồi của sinh viên Việt Nam. Mục tiêu chính của dự án là **so sánh hiệu quả giữa hai mô hình Transformer phổ biến**:

* **mBERT (Multilingual BERT)** – mô hình đa ngôn ngữ của Google
* **PhoBERT** – mô hình BERT được tối ưu riêng cho tiếng Việt

Thông qua quá trình **fine-tuning và đánh giá trên tập dữ liệu UIT-VSFC**, dự án phân tích sự khác biệt về hiệu năng giữa hai mô hình cũng như thực hiện **error analysis** để hiểu rõ các trường hợp mô hình dự đoán sai.

---

# Dataset

Dự án sử dụng bộ dữ liệu:

**UIT-VSFC – Vietnamese Students' Feedback Corpus**

Nguồn:
https://huggingface.co/datasets/ura-hcmut/UIT-VSFC

Bộ dữ liệu gồm các phản hồi của sinh viên về môn học và được gán nhãn cảm xúc:

* **Positive**
* **Neutral**
* **Negative**

Phân chia dữ liệu:

| Tập dữ liệu | Số lượng mẫu |
| ----------- | ------------ |
| Train       | 11,426       |
| Validation  | 1,584        |
| Test        | 3,166        |

---

# Các mô hình sử dụng

Hai mô hình Transformer được sử dụng để so sánh:

### 1. mBERT

* Model: `bert-base-multilingual-cased`
* Được huấn luyện trên hơn **100 ngôn ngữ**
* Sử dụng **WordPiece tokenization**

### 2. PhoBERT

* Model: `vinai/phobert-base`
* Được huấn luyện chuyên biệt cho **tiếng Việt**
* Yêu cầu **word segmentation trước khi tokenization**

Thư viện sử dụng:

* transformers
* datasets
* torch
* scikit-learn
* pyvi

---

# Cấu hình huấn luyện

Các mô hình được **fine-tune trong 3 epoch** với cấu hình giống nhau để đảm bảo so sánh công bằng.

| Tham số       | Giá trị             |
| ------------- | ------------------- |
| Epoch         | 3                   |
| Batch size    | 16                  |
| Learning rate | 2e-5                |
| Framework     | HuggingFace Trainer |

---

# Quy trình thực hiện

Pipeline của dự án gồm các bước chính:

1. **Cài đặt môi trường**

   * transformers
   * datasets
   * pyvi
   * torch

2. **Tải và khám phá dữ liệu**

   * tải dataset từ HuggingFace
   * kiểm tra phân bố nhãn

3. **Tokenization**

   * mBERT: WordPiece
   * PhoBERT: Word Segmentation + BPE

4. **Tiền xử lý dữ liệu**

   * ánh xạ nhãn
   * loại bỏ dữ liệu rỗng
   * chuyển sang tensor

5. **Huấn luyện mô hình**

   * fine-tune mBERT
   * fine-tune PhoBERT

6. **Đánh giá mô hình**

   * Accuracy
   * Precision
   * Recall
   * Macro F1-score
   * Confusion Matrix

7. **Error Analysis**

   * phân tích các câu bị phân loại sai
   * kiểm tra đặc điểm câu gây nhầm lẫn

---

# Kết quả mô hình

### mBERT

| Metric         | Giá trị    |
| -------------- | ---------- |
| Accuracy       | **0.9245** |
| Macro F1-score | **0.7952** |
| Eval Loss      | 0.3338     |
| Runtime        | 27.43s     |

---

### PhoBERT

| Metric         | Giá trị    |
| -------------- | ---------- |
| Accuracy       | **0.9413** |
| Macro F1-score | **0.8346** |
| Eval Loss      | 0.2867     |
| Runtime        | 23.27s     |

---

# So sánh mô hình

| Model   | Accuracy   | Macro F1   |
| ------- | ---------- | ---------- |
| mBERT   | 0.9245     | 0.7952     |
| PhoBERT | **0.9413** | **0.8346** |

Kết quả cho thấy:

* **PhoBERT vượt trội hơn mBERT** trên cả Accuracy và F1-score
* Điều này cho thấy **mô hình được huấn luyện riêng cho tiếng Việt** có lợi thế trong các bài toán NLP tiếng Việt.

---

# Phân tích lỗi (Error Analysis)

Sau khi đánh giá mô hình, dự án tiếp tục phân tích:

* Các câu bị dự đoán sai
* Các cặp nhãn bị nhầm lẫn nhiều nhất
* Ảnh hưởng của **độ dài câu** đến khả năng dự đoán

Một số nguyên nhân phổ biến gây lỗi:

* Câu quá ngắn
* Từ lóng của sinh viên
* Các câu mang tính **trung lập nhưng chứa từ mang cảm xúc**

---

# Cấu trúc dự án

```
project
│
└── BERTvsPhoBERT_for_VietNamese_text_classification.ipynb
```

Notebook được xây dựng và chạy trên **Google Colab**.

---

# Cách chạy dự án

### 1. Mở Google Colab

Upload hoặc mở notebook:

```
BERTvsPhoBERT_for_VietNamese_text_classification.ipynb
```

### 2. Cài đặt thư viện

```
pip install transformers datasets pyvi scikit-learn
```

### 3. Chạy toàn bộ notebook

Thực hiện lần lượt các cell để:

* tải dữ liệu
* tiền xử lý
* huấn luyện
* đánh giá mô hình

---

# Hướng phát triển trong tương lai

Một số hướng có thể cải thiện dự án:

* Huấn luyện nhiều epoch hơn (5–10 epoch)
* Thử nghiệm các mô hình mới:

  * viBERT
  * viDeBERTa
* Data augmentation cho tiếng Việt
* Hyperparameter tuning
* Sử dụng **cross-validation**

---

# Tài liệu tham khảo

PhoBERT:
https://arxiv.org/abs/2003.00744

HuggingFace Transformers:
https://huggingface.co/docs/transformers

UIT-VSFC Dataset:
https://huggingface.co/datasets/ura-hcmut/UIT-VSFC

---

# Ghi chú

Notebook được xây dựng nhằm mục đích:

* học tập về **Vietnamese NLP**
* thực hành **Transformer fine-tuning**
* so sánh mô hình **đa ngôn ngữ vs mô hình chuyên biệt**
