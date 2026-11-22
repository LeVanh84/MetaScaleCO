# MetaScaleCO

Slide và code prototype cho ý tưởng **MetaScale-CO**: học cách phân bổ test-time compute khi giải bài toán tối ưu tổ hợp. Thay vì chạy một lịch cố định, controller nhìn trạng thái tìm kiếm để chọn operator tiếp theo dưới một budget cho trước.

> **Phạm vi.** Ý tưởng lấy từ proposal MetaScale-CO. Code ở đây là prototype trên TSP, dùng **operator cổ điển chạy CPU** thay cho solver neural (GenSCO). Số liệu bên dưới là của prototype, không phải kết quả của proposal.

## Cấu trúc

```
slides/                     main.pdf (bản trình chiếu), main.tex và theme
code/
├── scripts/run_all.sh      chạy toàn bộ pipeline
├── config.json             seed, budget, tập dữ liệu, lưới tham số
└── metascale_co/
    ├── problems/           TSP: 2-opt, or-opt, destroy–repair, double-bridge
    ├── operators/          6 operator dùng chung một interface
    ├── environments/       môi trường tìm kiếm có budget (đơn lẻ và batch)
    ├── controllers/        state encoder, VoC model, MetaScale và các baseline
    ├── data/               sinh instance, lời giải tham chiếu LKH, trajectory log
    ├── training/           thu log, train VoC, tune tham số
    ├── evaluation/         so sánh, thống kê, vẽ biểu đồ
    └── methods.py          danh sách các phương pháp được so sánh
```

## Cài đặt

Cần [uv](https://docs.astral.sh/uv/getting-started/installation/) và bash. uv tự cài các thư viện trong `pyproject.toml` (Python 3.11 hoặc 3.12). Không cần GPU.

```bash
git clone https://github.com/LeVanh84/MetaScaleCO.git
cd MetaScaleCO/code
```

## Chạy

```bash
./scripts/run_all.sh
```

Mất khoảng 5 phút trên Apple M1 Pro (8 nhân, 16 GB RAM); chưa thử trên Windows và Linux. Chương trình in tiến độ ra terminal và lưu kết quả ra file, không mở cửa sổ nào.

Có thể chạy riêng từng bước, theo đúng thứ tự:

```bash
uv run python -m metascale_co.data.datasets               # 1. dữ liệu, LKH, đơn vị compute C0
uv run python -m metascale_co.training.collect_rollouts   # 2. thu log các lịch tìm kiếm
uv run python -m metascale_co.training.train_voc          # 3. train VoC model
uv run python -m metascale_co.training.tune               # 4. tune trên tập validation
uv run python -m metascale_co.evaluation.budget_sweep     # 5. so sánh 11 phương pháp
uv run python -m metascale_co.evaluation.batch_eval       # 6. nhiều bài chia chung budget
uv run python -m metascale_co.evaluation.stats            # 7. kiểm định thống kê
uv run python -m metascale_co.evaluation.plots            # 8. vẽ biểu đồ
```

## Xem kết quả

Các thư mục này được tạo khi chạy, nằm trong `code/`:

| File | Nội dung |
|---|---|
| `figures/fig1_quality_compute_scaling` | Gap theo budget của các phương pháp (hình trên slide) |
| `figures/fig3_compute_frontier_and_wallclock` | Learned STOP và wall-clock |
| `figures/fig5_voc_calibration` | Gain dự đoán so với gain thật |
| `figures/fig7_batch_allocation` | Budget gộp so với chia đều |
| `results/stats_summary.txt` | Tóm tắt thống kê, nên đọc trước (tạo khi chạy `run_all.sh`) |
| `results/results.csv` | Kết quả từng lượt chạy |

Mỗi hình có bản PDF và PNG. Trên macOS: `open figures/*.png`.

## Cách hoạt động

1. Dựng tour ban đầu bằng nearest neighbour + 2-opt.
2. Mã hoá state: budget còn lại, số bước bế tắc, hiệu quả gần đây của từng operator.
3. Chấm điểm từng operator bằng $U = \mu + \beta\sigma - \lambda\tilde c$ (gain dự đoán, độ không chắc, chi phí), rồi chạy operator có điểm cao nhất trong phạm vi budget.
4. Chỉ giữ tour mới nếu ngắn hơn, trừ chi phí khỏi budget, lặp lại tới khi hết budget.

Budget tính theo bội của $C_0$, là chi phí trung bình của một lần chạy operator `light`.

## Kết quả

Gap trung bình so với LKH (%), TSP-100 uniform, 100 instance × 3 seed, càng nhỏ càng tốt:

| Budget ($C_0$) | Fixed-light | Fixed-tuned | MetaScale-UCB-epi | Oracle một bước (cận tham khảo) |
|---|---|---|---|---|
| 2 | 4.946 | 4.123 | 4.273 | 3.585 |
| 16 | 3.098 | 2.075 | 2.030 | 1.363 |
| 128 | 1.485 | 0.582 | 0.548 | 0.193 |

Tính trên 21 ô (3 tập test × 7 budget), kiểm định Wilcoxon signed-rank, hiệu chỉnh Holm, α = 0.05:

- So với **Fixed-light**: MetaScale-UCB-epi tốt hơn ở 20 ô, 1 ô không khác biệt có ý nghĩa.
- So với **Fixed-tuned** (operator cố định tốt nhất cho từng budget): tốt hơn 8 ô, kém 2 ô, 11 ô không khác biệt có ý nghĩa. Post-hoc sau Friedman không cho thấy khác biệt ($p_{\mathrm{Holm}} = 0.24$).

## Biên dịch slide

Cần TeX Live có gói `vietnam`:

```bash
cd slides
latexmk -pdf main.tex
```
