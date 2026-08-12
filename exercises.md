# Day 14 — Exercises

## AI Evaluation & Benchmarking · Lab Worksheet

**Thời gian làm bài:** 14:15–17:00

**Domain:** OrbitTech Store Customer Support

Điền trực tiếp câu trả lời vào file này. Golden dataset 20 QA được viết một lần
duy nhất trong `golden_dataset.json`, không chép lại toàn bộ vào Markdown.

---

Từ 14:15–14:30, cài môi trường và chạy baseline tests theo `guide_lab.md`.

---

## Part 1 — Warm-up (14:30–14:45)

### Exercise 1.1 — RAGAS Metric Thresholds

Theo bài giảng:

- 0.8–1.0: Good — monitor, maintain.
- 0.6–0.8: Needs work — analyze failures, iterate.
- Dưới 0.6: Significant issues — investigate.

Với từng metric, xác định khi nào score thấp có thể chấp nhận và khi nào là
critical.

| Metric | Kịch bản điểm thấp có thể chấp nhận | Kịch bản điểm thấp nghiêm trọng | Hành động cần thực hiện |
|---|---|---|---|
| Faithfulness | Câu trả lời diễn giải đúng ý context nhưng dùng từ đồng nghĩa khác, khiến heuristic đếm token-overlap tính thiếu độ grounded thực tế (ví dụ 0.6–0.7 nhưng không có claim mâu thuẫn). | Câu trả lời nêu một chi tiết chính sách cụ thể (% hoàn tiền, thời hạn bảo hành, hạn đổi trả) hoàn toàn không có trong context đã retrieve — một claim bịa đặt, ảnh hưởng tiền bạc. | Nghiêm trọng → điều tra ngay: kiểm tra context đã retrieve và hướng dẫn grounding trong generation prompt. Faithfulness có mức độ nghiêm trọng cao nhất vì câu trả lời hỗ trợ của OrbitTech liên quan tới refund/warranty, nên hallucination có rủi ro tài chính/pháp lý trực tiếp. |
| Answer Relevance | Câu hỏi có nhiều ý nhỏ; agent trả lời đúng ý chính nhưng heuristic trừ điểm vì không lặp lại đúng token của câu hỏi. | Agent trả lời sai chủ đề hoàn toàn so với câu hỏi (ví dụ khách hỏi về quy trình bảo hành, answer lại nói về vận chuyển) — lỗi `off_topic`. | Nghiêm trọng → rà lại routing/prompt (nhận sai intent). Điểm thấp ở mức chấp nhận được → tinh chỉnh heuristic đo relevance thay vì đụng vào logic generation. |
| Context Recall | Câu hỏi thuộc Easy/một document; retriever lấy đúng chunk bằng chứng chính nhưng bỏ sót một chunk phụ, không thiết yếu. | Câu hỏi thuộc Hard/nhiều điều kiện (ví dụ có điều khoản ngoại lệ trong `09_escalation_and_policy_updates.md`); retriever bỏ sót hoàn toàn điều khoản ngoại lệ, khiến answer chỉ nêu quy tắc chung mà thiếu ngoại lệ — hậu quả thực tế cho khách hàng. | Nghiêm trọng → sửa retrieval (chunking, embedding model, query rewriting, top-k). Chấp nhận được → chỉ theo dõi, không cần chỉnh retriever. |
| Context Precision | Chunk đúng vẫn được retrieve nhưng xếp hạng 2–3 do một document liên quan có overlap ngữ nghĩa gần (ví dụ cả returns và warranty đều nhắc "30 ngày"). | Chunk đúng vẫn được retrieve nhưng bị chôn gần cuối top-k, trong khi nhiều chunk không liên quan xếp hạng cao hơn — đẩy bằng chứng gold ra khỏi vùng chú ý hiệu quả của generator dù recall vẫn ổn. | Nghiêm trọng → rerank hoặc tinh chỉnh embedding/query. Đây thường là nguyên nhân "ngầm" khiến Faithfulness giảm dù Recall một mình vẫn có vẻ ổn. |
| Completeness | Expected answer có một chi tiết phụ (ví dụ một khuyến mãi liên quan) bị bỏ sót, nhưng mọi bước hành động cốt lõi vẫn đầy đủ. | Case Hard yêu cầu nhiều điều kiện; agent chỉ đáp ứng một trong số nhiều điều kiện bắt buộc (ví dụ bỏ sót ngoại lệ cho hàng điện tử đã mở hộp trong chính sách đổi trả) — hướng dẫn gây hiểu lầm. | Nghiêm trọng → kiểm tra xem Context Recall có thấp không (do retrieval) hay context đã đủ nhưng generation tổng hợp kém (do generation), rồi sửa tương ứng. |

### Exercise 1.2 — Bias trong LLM-as-a-Judge

Ba bias thường gặp:

- Position bias: judge ưu tiên answer xuất hiện trước.
- Verbosity bias: judge ưu tiên answer dài hơn.
- Self-preference: judge ưu tiên output giống chính model đó.

**Câu 1: Thiết kế experiment phát hiện position bias với ít nhất hai conditions.**

> *Câu trả lời:* Chuẩn bị N cặp câu trả lời (A, B) có chất lượng tương đương đã được xác nhận độc lập (bằng human label hoặc một judge khác mạnh hơn), để bất kỳ sự thiên vị nào của judge chỉ có thể do vị trí, không phải do chất lượng. Chạy judge hai lần trên cùng một cặp, giữ nguyên prompt/rubric/temperature, chỉ đổi thứ tự hiển thị:
> - **Condition 1:** hiển thị A trước, B sau.
> - **Condition 2:** hiển thị B trước, A sau.
>
> Nếu judge có xu hướng chọn "câu xuất hiện trước" nhiều hơn 50% một cách có ý nghĩa thống kê khi tổng hợp trên toàn bộ N cặp (dùng sign test hoặc McNemar's test), đó là bằng chứng của position bias. Có thể thêm condition thứ ba là random hóa thứ tự trên nhiều lần lặp để tính effect size chính xác hơn (chênh lệch win-rate giữa slot đầu và slot sau cho cùng một answer, trung bình qua nhiều cặp).

**Câu 2: Làm thế nào giảm verbosity bias bằng rubric design?**

> *Câu trả lời:*
> - Viết rubric theo tiêu chí nội dung cụ thể (ví dụ "trả lời đúng đủ 3/3 điều kiện", "có trích dẫn đúng document") thay vì dùng từ mơ hồ như "chi tiết" hoặc "đầy đủ" — những từ này dễ bị judge đánh đồng với "dài".
> - Thêm dòng rubric tường minh: một câu trả lời ngắn nhưng đủ nội dung bắt buộc phải được chấm ngang bằng một câu dài hơn; lặp lại hoặc thêm nội dung thừa không cần thiết phải bị trừ điểm, không được thưởng điểm.
> - Cung cấp few-shot calibration examples trong đó một câu trả lời ngắn, đúng được chấm cao hơn một câu trả lời dài nhưng thừa thãi, để anchor judge trước khi chấm thật.
> - Tách riêng dimension "Conciseness/Clarity" khỏi "Completeness", để judge không nhầm lẫn giữa "dài" và "đầy đủ".

**Câu 3: Tại sao cần calibrate LLM judge với human labels?**

> *Câu trả lời:* LLM judge không tự động đảm bảo align với đánh giá của con người hoặc domain expert — nó có thể mang các bias hệ thống (position, verbosity, self-preference) hoặc hiểu sai các nuance domain-specific (ví dụ policy có `effective_date` trong `09_escalation_and_policy_updates.md`). Calibration — so sánh judge score với một tập nhỏ human-labeled ground truth và đo độ đồng thuận (Cohen's kappa hoặc correlation) — giúp: (1) phát hiện judge quá khắt khe hoặc quá dễ dãi một cách hệ thống, (2) điều chỉnh threshold/rubric cho khớp kỳ vọng thực tế của business, (3) tạo confidence để dùng automated evaluation thay thế một phần human review trong CI/CD mà không mất độ tin cậy.

### Exercise 1.3 — Evaluation trong CI/CD

**Câu 1: Chọn threshold để block deployment.**

| Metric | Threshold | Lý do |
|---|---:|---|
| Faithfulness | ≥ 0.75 | Support answer liên quan trực tiếp tới tiền và chính sách (refund, warranty), nên hallucination có hệ quả tài chính/pháp lý cao nhất trong bốn metric. Threshold đặt gần vùng "Good" (0.8) nhưng chừa biên độ nhỏ cho nhiễu của heuristic. |
| Answer Relevance | ≥ 0.70 | Trả lời lạc đề gây trải nghiệm xấu và tăng chi phí hỗ trợ (khách phải hỏi lại) nhưng rủi ro pháp lý thấp hơn faithfulness, nên threshold thấp hơn một bậc, đúng ranh giới "Needs work" theo thang điểm bài giảng. |
| Completeness | ≥ 0.65 | Thiếu một chi tiết phụ có thể chấp nhận tạm thời và không nên chặn deploy vì rủi ro thấp nhất trong ba metric; threshold thấp hơn để tránh false-positive block do các case Hard tự nhiên có completeness thấp hơn. |

**Câu 2: Khi nào dùng offline evaluation, online evaluation và human review?**

> *Câu trả lời:*
> - **Offline:** mỗi lần release hoặc thay đổi prompt/model/retriever, chạy trên golden dataset cố định (20 QA) để so sánh regression trước khi merge/deploy — dùng `run_regression()` để phát hiện metric giảm quá 0.05 so với baseline.
> - **Online:** continuous monitoring trên real traffic sau khi đã deploy, để phát hiện drift khi corpus/policy cập nhật (ví dụ `effective_date` mới trong `09_escalation_and_policy_updates.md`) hoặc khi user hỏi những câu ngoài phạm vi golden dataset — dùng tracing/feedback functions như TruLens hoặc Langfuse.
> - **Human review:** dành cho high-stakes cases (account security, payment fraud, khiếu nại warranty lớn), khi automated judge score nằm trong vùng biên/không chắc chắn, hoặc định kỳ để calibrate LLM judge; cũng cần cho các tiêu chí subjective như tone/empathy mà heuristic hoặc LLM judge chưa đủ tin cậy để tự động hóa hoàn toàn.

---

## Part 2 — Core Coding (14:45–15:40)

Hoàn thiện các TODO bắt buộc trong `template.py`.

### Task 1 — Data Models

- `QAPair`: question, expected answer, gold context, metadata và retrieved contexts.
- `EvalResult`: answer-side scores, optional retrieval scores, pass/failure fields.
- `overall_score()`: trung bình Faithfulness, Relevance và Completeness.

### Task 2 — RAGASEvaluator

Answer-side:

- `evaluate_faithfulness(answer, context)`
- `evaluate_relevance(answer, question)`
- `evaluate_completeness(answer, expected)`

Retrieval-side:

- `evaluate_context_recall(contexts, expected)`
- `evaluate_context_precision(contexts, expected)`

Full pipeline:

- `run_full_eval(..., contexts=None)` luôn tính ba answer metrics.
- Nếu có `contexts`, tính và lưu thêm Context Recall và Context Precision.
- Retrieval scores không làm thay đổi `overall_score()` và pass rule gốc.

### Task 3 — LLMJudge

- `score_response(question, answer, rubric)`
- `detect_bias(scores_batch)`

### Task 4 — BenchmarkRunner

- `run(qa_pairs, agent_fn, evaluator)`
- `generate_report(results)`
- `run_regression(new_results, baseline_results)`
- `identify_failures(results, threshold)`

`BenchmarkRunner.run()` phải truyền `pair.retrieved_contexts` vào
`run_full_eval()`. Report phải có average của hai retrieval metrics.

### Task 5 — FailureAnalyzer

- `categorize_failures(failures)`
- `find_root_cause(failure)`
- `generate_improvement_suggestions(failures)`
- `generate_improvement_log(failures, suggestions)`

Kiểm tra:

```bash
pytest tests/ -v
```

`rerank_by_overlap()` là TODO bonus của Exercise 3.5. Test tương ứng được skip
nếu bạn chưa làm bonus.

---

## Part 3 — Golden Dataset & Real Benchmark (15:40–16:35)

### Exercise 3.1 — Build the Golden Dataset

Thiết kế và validate dataset theo Mục 5–6 trong `guide_lab.md`. Nội dung 20 QA
được điền trực tiếp trong `golden_dataset.json`; phần dưới chỉ ghi lại kết quả
và quyết định thiết kế, không chép lại toàn bộ QA.

**Kết quả dataset**

| Hạng mục | Kết quả |
|---|---|
| Tổng số records | ____ / 20 |
| Easy | ____ / 5 |
| Medium | ____ / 7 |
| Hard | ____ / 5 |
| Adversarial | ____ / 3 |
| Source documents được sử dụng | ____ / 10 |
| Validator status | PASS / FAIL |

**Ba case đại diện cho quyết định thiết kế**

| ID | Difficulty | Source document(s) | Vì sao case phù hợp với difficulty/attack type? |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

**Điểm khó nhất khi xây dựng expected answer hoặc evidence là gì?**

> *Câu trả lời:*

**Xác nhận:**

- [ ] Mọi claim trong expected answer đều có evidence hỗ trợ.
- [ ] Không có questions trùng ý và không dùng kiến thức ngoài corpus.
- [ ] `python validate_golden_dataset.py` báo `PASS`.

### Exercise 3.2 — Benchmark Run

Chạy:

```bash
python domain_assistant.py
python evaluate_answers.py
```

Copy bảng terminal vào đây hoặc điền từ `artifacts/benchmark_results.json`.

| ID | Question (short) | Ctx Recall | Ctx Precision | Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| E01 | | | | | | | | | |
| E02 | | | | | | | | | |
| E03 | | | | | | | | | |
| E04 | | | | | | | | | |
| E05 | | | | | | | | | |
| M01 | | | | | | | | | |
| M02 | | | | | | | | | |
| M03 | | | | | | | | | |
| M04 | | | | | | | | | |
| M05 | | | | | | | | | |
| M06 | | | | | | | | | |
| M07 | | | | | | | | | |
| H01 | | | | | | | | | |
| H02 | | | | | | | | | |
| H03 | | | | | | | | | |
| H04 | | | | | | | | | |
| H05 | | | | | | | | | |
| A01 | | | | | | | | | |
| A02 | | | | | | | | | |
| A03 | | | | | | | | | |

**Aggregate Report**

- Overall pass rate: ____%
- Avg Context Recall: ____
- Avg Context Precision: ____
- Avg Faithfulness: ____
- Avg Relevance: ____
- Avg Completeness: ____
- Failure type distribution: ____

**Ba cases có Overall Score thấp nhất**

1. ID: ____ | Score: ____ | Failure type: ____
2. ID: ____ | Score: ____ | Failure type: ____
3. ID: ____ | Score: ____ | Failure type: ____

**Nhận xét ngắn:** Metric nào yếu nhất? Kết quả gợi ý vấn đề nằm ở retrieval
hay generation?

> *Câu trả lời:*

### Exercise 3.3 — LLM-as-a-Judge Rubric Design

Thiết kế rubric domain-specific cho OrbitTech Customer Support. Mỗi mức phải
đủ cụ thể để hai người chấm độc lập có thể hiểu giống nhau.

Chọn 3–5 dimensions:

- [ ] Correctness
- [ ] Completeness
- [ ] Relevance
- [ ] Evidence/citation
- [ ] Actionability
- [ ] Safety/privacy
- [ ] Tone/clarity
- [ ] Dimension khác: __________

| Score | Tiêu chí domain-specific | Ví dụ response |
|---:|---|---|
| 5 | | |
| 4 | | |
| 3 | | |
| 2 | | |
| 1 | | |

**Ba edge cases khó chấm**

| Edge Case | Tại sao khó chấm? | Rubric xử lý thế nào? |
|---|---|---|
| | | |
| | | |
| | | |

**Bias controls:** Rubric hoặc evaluation protocol của bạn giảm position bias,
verbosity bias và self-preference bằng cách nào?

> *Câu trả lời:*

### Exercise 3.4 — Framework Comparison (Bonus +10)

Chỉ làm sau khi hoàn thành 3.1–3.3. Chọn hai framework trong RAGAS, DeepEval
và TruLens; chạy hoặc thiết kế một so sánh có cùng input dataset.

| Tiêu chí | Framework 1: ____ | Framework 2: ____ |
|---|---|---|
| Setup complexity | | |
| Metrics available | | |
| CI/CD integration | | |
| Kết quả trên cùng dataset | | |
| Insight rút ra | | |

- Scores có nhất quán không?
- Framework nào strict hơn và vì sao?
- Hai framework có tìm ra cùng failure cases không?

> *Phân tích:*

### Exercise 3.5 — Retrieval Reranking (Bonus +5)

Mục tiêu: kiểm tra việc đổi thứ tự chunks có tăng Context Precision mà không
thay đổi Context Recall hay không.

1. Chọn ít nhất 5 cases từ `artifacts/actual_answers.json`.
2. Tính Context Recall và Context Precision trước rerank.
3. Implement `rerank_by_overlap()` hoặc một reranker khác.
4. Rerank cùng tập chunks, không thêm hoặc xóa chunk.
5. Tính lại hai metrics và giải thích kết quả.

| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |
|---|---:|---:|---:|---:|---:|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| **Avg** | | | | | |

**Tại sao Recall dự kiến không đổi?**

> *Câu trả lời:*

**Khi nào reranking không đủ và cần sửa retriever/query/chunking?**

> *Câu trả lời:*

---

## Part 4 — Reflection (16:35–16:50)

Hoàn thành `reflection.md` bằng kết quả thật từ Exercise 3.2.

---

## Completion Checklist

Hoàn thành kiểm tra cuối trong khoảng 16:50–17:00.

- [ ] Tất cả required tests pass.
- [ ] `golden_dataset.json` validate thành công.
- [ ] Exercise 3.1 hoàn thành trong file JSON và bảng kết quả phía trên.
- [ ] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.
- [ ] Exercise 3.3 có rubric 1–5 và bias controls.
- [ ] `reflection.md` có ba failure analyses và regression strategy.
- [ ] Đã copy `template.py` thành `solution/solution.py`.
- [ ] Exercise 3.4 và 3.5 chỉ làm nếu chọn bonus.
