"""复现 Bocharov & Thiesson (ICPRAM 2012 / AISC 204, 2013)：基于谱聚类的斜切分 ART（自回归树）模型。

模块：
- bayes.py      叶子线性回归的共轭 Normal-Inverse-Gamma 边际似然与后验预测（替代 Meek et al. 2002 的贝叶斯评分，见 notes）
- spectral.py   论文 Fig.1 的简化归一化谱聚类（第二大特征向量 + 最大间隙切分）+ 口袋感知机
- art.py        切分提议器（AxialGaussian / ObliqueSpectral / Fixed / TimeSpectral）、贪心建树、逐阶加入自回归项、序贯预测得分
- synthetic.py  合成状态切换序列验证
- sp500.py      在本项目 S&P 500 日线上的近似复现实验（价格水平，年块切分协议）
- returns_exp.py 日对数收益率版本（含 ZeroMean/ConstMean 参考与波动率 regime 诊断）
- cli.py        命令行入口 paper-repro

依赖方向：paper_repro 可导入 trend_scan（路径、价格缓存），trend_scan 不导入 paper_repro。
"""
