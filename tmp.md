Mô hình	Complete Macro-F1 $\uparrow$	Complete Micro-F1 $\uparrow$	Hamming Accuracy $\uparrow$	Selective Macro-F1 ($c=0.30$) $\uparrow$	Coverage ($c=0.30$) $\uparrow$	Generalized Loss $\downarrow$	Error Capture Rate (ECR) $\uparrow$
BR_MLP	0.3251	0.3710	0.7731	(Không từ chối)	1.0000	0.2269	—
CC_MLP	0.3775	0.5558	0.8569	(Không từ chối)	1.0000	0.1431	—
MLC_PA_MLP	0.3251	0.3710	0.7731	0.2107	0.0307 (Sụp đổ)	0.2917	0.9961
GSI_MLC_PA_MLP (Ours)	0.3377	0.4400	0.8362	0.3520	0.5311	0.1525	0.6925


Tập dữ liệu	MLC-PA Macro-F1	GSI Macro-F1	Mức tăng ($\Delta$)	MLC-PA Coverage	GSI Coverage	GSI Generalized Loss $\downarrow$
emotions	0.3667	0.5297	+0.1630	1.48%	26.70%	0.2368
music	0.4430	0.5453	+0.1023	2.96%	27.75%	0.2356
scene	0.9324	0.7435	-0.1889 (đổi lại Coverage gấp 4)	11.13%	43.90%	0.1740 (thấp hơn MLC-PA)
yeast	0.2541	0.4607	+0.2066	1.41%	12.78%	0.2673
genbase	0.0000	0.6305	+0.6305	2.22%	99.72%	0.0040
medical	0.0089	0.1806	+0.1717	4.46%	97.57%	0.0193
cal500	0.0011	0.0827	+0.0815	0.01%	29.45%	0.2490
enron	0.0777	0.1668	+0.0891	1.29%	71.28%	0.1001
reuters-k500	0.0000	0.0941	+0.0941	4.79%	52.54%	0.1438
bibtex	0.0226	0.0865	+0.0639	0.93%	69.41%	0.0953
TRUNG BÌNH	0.2107	0.3520	+0.1413 (+14.13%)	3.07%	53.11% (x17.3 lần)	0.1525 (-47.7%)


Chỉ số đánh giá	GSI-MLC-PA Cũ (Hamming BOP)	GSI-MLC-PA Mới (Per-Label Macro-F1)	Mức độ cải thiện
Selective Macro-F1	0.2115	0.3520	Tăng +14.05% tuyệt đối (+66.4% tương đối)
Coverage ($c=0.30$)	41.95%	53.11%	Tăng +11.16% độ phủ quyết định
Generalized Loss	0.1839	0.1525	Giảm 17.1% tổn thất tổng quát
