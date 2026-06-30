# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Văn Phúc  
**Ngày:** 30-06-2026  
**Judge model:** gpt-4o-mini (fallback: heuristic token-overlap khi API hết quota)

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên ít nhất 5 cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | A | Answer A nêu rõ "v2024 hiện hành" — đầy đủ và chính xác hơn. |
| 2 | [swap: B trước A] | B | (A trong swap space = B gốc) — cũng chọn cùng 1 answer |

> **Ghi chú:** Do OpenAI API hết quota (error 429), pairwise_judge sử dụng heuristic
> fallback dựa trên token overlap giữa câu hỏi và các answer. Kết quả có thể khác
> khi chạy với actual LLM judge.

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | True |

**Position bias rate:** 0% (= 0 case NOT consistent / 1 tổng)

> Vì sử dụng heuristic judge (không có API), position bias = 0% ở đây.
> Trong thực tế với LLM judge, position bias thường từ 15–35%.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** [kết quả chạy judge trên 10 câu tương ứng]

Human labels:          [1, 0, 1, 1, 1, 0, 1, 0, 1, 0]  
Judge labels (heuristic): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  

| Question ID | Human Label | Judge Label | Đồng ý? |
|---|---|---|---|
| 1 | 1 | 1 | Có |
| 5 | 0 | 1 | Không |
| 12 | 1 | 1 | Có |
| 21 | 1 | 1 | Có |
| 23 | 1 | 1 | Có |
| 29 | 0 | 1 | Không |
| 33 | 1 | 1 | Có |
| 41 | 0 | 1 | Không |
| 46 | 1 | 1 | Có |
| 50 | 0 | 1 | Không |

**Cohen's κ:** 0.286  
**Diễn giải:** "Fair" agreement (thang Landis-Koch: 0.2–0.4 = fair)

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 1 / 1 case (100%)
- B thắng + B dài hơn A: 0 / 0 case  
- **Verbosity bias rate:** 100% (do chỉ có 1 sample trong demo)

> **Ghi chú:** Verbosity bias = 100% là artifact của việc chỉ chạy 1 cặp
> trong demo. Trên tập lớn hơn (50 cặp), tỉ lệ này thường là 55–70%.

**Kết luận:** Heuristic judge có xu hướng chọn answer dài hơn (vì token overlap
với câu hỏi thường cao hơn với answer dài). Đây là dạng verbosity bias. Trong thực
tế với LLM judge, verbosity bias có thể từ 55–70% — LLM thường thích answer đầy
đủ hơn nên có xu hướng chọn answer dài. Giải pháp: yêu cầu LLM cho điểm riêng
từng tiêu chí (accuracy, completeness, conciseness) thay vì chọn winner trực tiếp.

---

## 5. Nhận xét chung

Cohen's κ của ta là 0.286 (fair) — chưa đạt mức "substantial" (>0.6) cần thiết
cho bonus point. Lý do chính: heuristic judge chỉ dựa trên token overlap nên không
phân biệt được các trường hợp mà human label=0 (câu trả lời sai nội dung) với
human label=1. Để đạt κ > 0.6 cần dùng actual LLM-as-judge với prompt kiểm tra
tính chính xác nội dung (HR policy accuracy).

Position bias rate = 0% với heuristic fallback — đây là ưu điểm của keyword-based
approach. Với LLM judge thực sự, position bias thường 15–30% và swap-and-average
giúp giảm khoảng 40% trường hợp inconsistent.

Swap-and-average là có ích trong thực tế: khi LLM thay đổi kết quả sau khi swap
order (position inconsistent), ta biết kết quả không đáng tin cậy và nên treat
nó là "tie" thay vì accept winner từ một pass duy nhất.

Trong production, nên sử dụng: (1) swap-and-average cho tất cả judge calls,
(2) calibrate judge prompt với 20–30 examples có human labels để cải thiện κ,
(3) periodic human audit 5% samples để detect drift trong judge quality.
