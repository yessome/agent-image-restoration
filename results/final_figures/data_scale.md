# 数据规模与计数口径

对应[六场景结果](table2_13tool.md)、[组件消融](table6_ablation.md)及[噪声](noise_strength_curve.png)和[低光](lowlight_strength_curve.png)强度曲线。

## 一、数据拆分（唯一源图数）

| 划分 | 源图数 | 用途 |
|---|---:|---|
| train | 1,765 | router 训练（抽样 360 组 / 1,080 个退化样本） |
| valA | 165 | 开发划分 |
| valB | 164 | 开发划分：消融与探索性验证 |
| test | 430 | 正式评测 |

去重后共 2,524 张源图（原 2,527，剔除 3 组字节级重复）。

## 二、test 六场景：2,580 个场景实例

**430 张唯一源图 × 6 个场景 = 2,580 个场景实例（每个 seed）**。

每个场景使用同一批 430 张源图。

| 场景 | 张数 | 退化 |
|---|---:|---|
| Clean | 430 | 无 |
| Noise, sigma 25 | 430 | AWGN σ=25 |
| Low light, gamma 3.0 | 430 | γ=3.0 |
| Motion blur | 430 | 15 px 方向核，45° |
| JPEG, quality 30 | 430 | JPEG q30 |
| Mixed | 430 | γ=3.0 → σ=25 |

## 三、方法记录数

每个场景实例比较 **7 种方法**（raw / fixed denoise / fixed low light / diagnostic rule /
legacy rule agent / learned iterative agent / 内部 fixed_type 记账列）。

- 每个 seed：2,580 × 7 = **18,060 条方法记录**
- 三个 seed 合计：**54,180 条**

## 四、其他实验的规模

| 实验 | 规模 | 说明 |
|---|---|---|
| 强度曲线（7 档） | 7 × 430 = **3,010 张图/seed**；3 seed = 9,030 | 每档 430 张，逐档独立评测 |
| valB 消融 | **328 张**（低光 164 + 噪声 164），3 seed = 984 | 开发划分，非 test |
| replay 等价性核对 | **2,580 条** = 860 个场景实例 × 3 seed | 每个 seed 含低光和噪声各 430 个实例 |
