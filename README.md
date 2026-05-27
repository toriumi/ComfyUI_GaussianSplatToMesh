# ComfyUI_GaussianSplatToMesh

Gaussian Splat（PLY_DATA）からメッシュ（TRIMESH）への変換を行うComfyUIカスタムノードです。

## 概要

HY-World 2.0（WorldMirror）などで生成されたGaussian Splatting点群データ（`PLY_DATA`）を、3Dメッシュ（`TRIMESH`）に変換します。変換後のメッシュは `Hy3DExportMesh` ノードに接続して GLB/OBJ/PLY/STL 形式でエクスポートできます。

Open3Dに依存せず、scipy + trimesh + sklearn のみで動作するため、Python 3.13環境でも利用可能です。

## 対応するComfyUIノード

| ノード名 | 説明 |
|----------|------|
| `VNCCS_WorldMirrorV2_3D` | HY-World 2.0 シングルイメージ3D再構築（PLY_DATA出力） |
| `GaussianSplatToMesh` | **本ノード** — PLY_DATA → TRIMESH変換 |
| `Hy3DExportMesh` | TRIMESH → GLB/OBJ/PLY/STLエクスポート |

## ノード情報

- **ノード名**: `GaussianSplatToMesh`
- **表示名**: 🔷 Gaussian Splat to Mesh
- **カテゴリ**: `3D/mesh`

## 入力

### 必須入力

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `ply_data` | `PLY_DATA` | Gaussian Splat点群データ（VNCCS_WorldMirrorV2_3Dの出力） |

### オプション入力

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `method` | 選択 | `alpha_shape` | サーフェス再構築手法（`alpha_shape` / `ball_pivoting` / `marching_cubes`） |
| `alpha` | FLOAT | `0.0` | alpha_shape手法のアルファ値。0=凸包、大きい値=より詳細（0.5〜10.0推奨） |
| `resolution` | INT | `128` | marching_cubes手法のボクセルグリッド解像度（32〜512） |
| `radius_factor` | FLOAT | `1.5` | ball_pivoting手法の半径倍率（0.5〜10.0） |
| `max_points` | INT | `100000` | 最大点数（超過時はダウンサンプリング） |
| `remove_outliers` | BOOLEAN | `True` | 再構築前に統計的外れ値を除去 |
| `outlier_std_ratio` | FLOAT | `2.0` | 外れ値除去の標準偏差比率（小さい値=より積極的に除去） |

## 出力

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `trimesh` | `TRIMESH` | 変換されたメッシュオブジェクト（trimesh.Trimesh） |

## サーフェス再構築手法

### alpha_shape（推奨）
Delaunay三角形分割ベースのアルファシェイプ。高速で密な点群に適しています。
- `alpha=0`: 凸包（最も単純）
- `alpha>0`: 値が大きいほど詳細な形状を再現（穴も増える）

### ball_pivoting
近似ボールピボッティング。局所的なDelaunay三角形分割パッチを使用し、エッジ長でフィルタリングします。

### marching_cubes
ボリューメトリック再構築。点群をボクセルグリッドに変換し、マーチングキューブ法でメッシュ化します。滑らかな結果が得られますが、scikit-imageが必要です。

## ワークフロー接続例

```
[Load Image] → [VNCCS_WorldMirrorV2_3D] → (PLY_DATA) → [GaussianSplatToMesh] → (TRIMESH) → [Hy3DExportMesh] → GLB/OBJ
```

典型的なワークフロー:
1. 画像を `VNCCS_WorldMirrorV2_3D` に入力して3D再構築
2. 出力の `PLY_DATA` を `GaussianSplatToMesh` に接続
3. 変換された `TRIMESH` を `Hy3DExportMesh` に接続してエクスポート

## 依存パッケージ

| パッケージ | 用途 | 備考 |
|-----------|------|------|
| `numpy` | 数値計算 | ComfyUI環境に標準搭載 |
| `torch` | テンソル処理 | ComfyUI環境に標準搭載 |
| `scipy` | Delaunay三角形分割、KDTree、ガウシアンフィルタ | ComfyUI環境に標準搭載 |
| `trimesh` | メッシュ生成・処理 | ComfyUI環境に標準搭載 |
| `scikit-image` | marching_cubes手法で使用（オプション） | alpha_shape/ball_pivoting使用時は不要 |

> **注意**: Open3Dは不要です。Python 3.13でOpen3Dが利用できない問題を回避するため、scipy/trimeshベースで実装しています。

## インストール

### ComfyUI-Managerから
ComfyUI-Managerの「Install Custom Nodes」から `ComfyUI_GaussianSplatToMesh` を検索してインストール。

### 手動インストール
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/toriumi/ComfyUI_GaussianSplatToMesh.git
```

追加の依存パッケージのインストールは通常不要です（ComfyUI環境に含まれています）。

## ライセンス

MIT License
