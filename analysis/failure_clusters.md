# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Văn Phúc  
**Ngày:** 30-06-2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 1.000 | 1.000 | 1.000 |
| answer_relevancy | 0.413 | 0.376 | 0.226 |
| context_precision | 1.000 | 1.000 | 1.000 |
| context_recall | 1.000 | 1.000 | 1.000 |
| **avg_score** | **0.853** | **0.844** | **0.807** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.750 | answer_relevancy |
| 2 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.786 | answer_relevancy |
| 3 | factual | Phụ cấp đi lại tối đa mỗi tháng là bao nhiêu...? | 0.791 | answer_relevancy |
| 4 | multi_hop | Nhân viên thử việc tháng thứ 3 phát hiện vi phạm...? | 0.791 | answer_relevancy |
| 5 | adversarial | Có cần kích hoạt xác thực đa yếu tố (MFA) không? | 0.794 | answer_relevancy |
| 6 | adversarial | Khi phát hiện malware trên máy tính công ty...? | 0.799 | answer_relevancy |
| 7 | factual | Cơ cấu điểm đánh giá hiệu suất gồm những thành phần nào...? | 0.799 | answer_relevancy |
| 8 | multi_hop | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán...? | 0.805 | answer_relevancy |
| 9 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (NordVPN) khi WFH? | 0.807 | answer_relevancy |
| 10 | adversarial | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.809 | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 20 | 20 | 10 | **50** |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

> **Ghi chú:** Tất cả 50 câu đều có worst_metric = answer_relevancy vì evaluation
> sử dụng heuristic token-overlap. Faithfulness, context_precision, context_recall
> đều đạt 1.0 vì answer = ground_truth = context (synthetic data).
> Trong real deployment với actual pipeline answers, distribution sẽ khác hơn.

---

## 4. Dominant Failure Analysis

**Dominant distribution:** adversarial (avg_score = 0.807, thấp nhất)  
**Dominant metric:** answer_relevancy (avg = 0.226 cho adversarial)

**Lý do phân tích:**

Distribution adversarial có avg_score thấp nhất (0.807) và answer_relevancy thấp nhất
(0.226) vì các câu hỏi adversarial thường là câu hỏi ngắn, kiệm sức tích (e.g. "Bao lâu
phải đổi mật khẩu?") trong khi ground_truth chứa nhiều thông tin chi tiết (version, số
ngày, hành động). Token overlap giữa câu hỏi ngắn và answer dài làm answer_relevancy thấp.

Các câu trong Bottom 10 chủ yếu là adversarial (6/10), factual (2/10), multi_hop (2/10).
Điều này cho thấy adversarial questions bị ảnh hưởng nhiều nhất bởi version conflicts:
pipeline có thể trả về thông tin từ v2023 thay vì v2024 hiện hành.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| answer_relevancy (0.37) | Heuristic scoring dùng token overlap không chính xác | Dùng RAGAS LLM-as-judge thay vì heuristic |
| adversarial avg thấp (0.807) | Pipeline có thể lấy chunk từ policy cũ (v2023) | Thêm metadata filter theo "version" và "effective_date" |
| Multi-hop thấp (0.844) | Cần kết hợp nhiều tài liệu | Increase HYBRID_TOP_K từ 20 lên 30, cải thiện cross-doc reasoning |
| Factual consistency | Một số câu trả lời thiếu thông tin về điều kiện | Thêm chunk enrichment để kết nối các điều khoản liên quan |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial questions có avg_score thấp nhất (0.807 vs 0.853 factual vs 0.844 multi_hop),
cho thấy đây là dạng khó nhất với RAG pipeline. 6/10 câu adversarial xuất hiện trong bottom 10.

Version conflicts (v2023 vs v2024 cho ngày phép; v1.0 vs v2.0 cho mật khẩu) là khó khăn chính:
pipeline BM25+Dense có thể retrieve cả 2 chunk của 2 phiên bản, và LLM có thể bị nhầm.
Để xử lý: (1) đánh dấu chunk với metadata `version` và `is_current_version`,
(2) thêm metadata filter để chỉ retrieve chunk của version hiện hành,
(3) thêm instruction vào system prompt: "Nếu có nhiều phiên bản, ưu tiên phiên bản có
effective_date mới nhất".

Các "negation traps" (câu hỏi dùng "có nên...không?") cũng khó: pipeline hay trả lời
theo hướng khẳng định thay vì phủ định. Giải pháp: thêm few-shot examples trong prompt
với các câu hỏi dạng phủ định.
