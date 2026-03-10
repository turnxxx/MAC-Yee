好的，我现在已经完整阅读了论文和代码。下面结合论文的数学背景，详细解释线性求解器的配置和求解过程。
一、论文背景：对偶场 MAC-Yee 格式
你的论文求解的是 Navier-Stokes 方程的速度-涡度-压力对偶场形式。连续系统是：
1-形式方程组 (半整数时间步)：
∂
u
1
∂
t
+
ω
1
×
u
1
+
1
R
e
∇
×
ω
2
+
∇
P
0
=
f
∂t
∂u 
1
​
 
​
 +ω 
1
​
 ×u 
1
​
 + 
Re
1
​
 ∇×ω 
2
​
 +∇P 
0
​
 =f
∇
×
u
1
−
ω
2
=
0
,
∇
⋅
u
1
=
0
∇×u 
1
​
 −ω 
2
​
 =0,∇⋅u 
1
​
 =0
2-形式方程组 (整数时间步)：
∂
u
2
∂
t
+
ω
2
×
u
2
+
1
R
e
∇
×
ω
1
+
∇
P
3
=
f
∂t
∂u 
2
​
 
​
 +ω 
2
​
 ×u 
2
​
 + 
Re
1
​
 ∇×ω 
1
​
 +∇P 
3
​
 =f
∇
×
u
2
−
ω
1
=
0
,
∇
⋅
u
2
=
0
∇×u 
2
​
 −ω 
1
​
 =0,∇⋅u 
2
​
 =0
离散化后，使用 de Rham 复形在 DMStag 交错网格上布置自由度：
\(V^0_h\)（顶点）：压力 
P
0
P 
0
​
 
\(V^1_h\)（棱）：速度 
u
1
u 
1
​
 、涡量 
ω
1
ω 
1
​
 
\(V^2_h\)（面）：速度 
u
2
u 
2
​
 、涡量 
ω
2
ω 
2
​
 
\(V^3_h\)（单元中心）：压力 
P
3
P 
3
​
 
二、每个时间步需要求解的线性系统结构
时间离散采用 leapfrog 型交错时间步进（论文 Section 5）。每步需解一个鞍点系统。以 1-形式子步为例，隐式离散后的线性系统为：
[
A
u
u
A
u
ω
A
u
p
A
ω
u
A
ω
ω
0
A
p
u
0
0
]
[
u
1
ω
2
P
0
]
=
[
rhs
u
rhs
ω
0
]
​
  
A 
uu
​
 
A 
ωu
​
 
A 
pu
​
 
​
  
A 
uω
​
 
A 
ωω
​
 
0
​
  
A 
up
​
 
0
0
​
  
​
  
​
  
u 
1
​
 
ω 
2
​
 
P 
0
​
 
​
  
​
 = 
​
  
rhs 
u
​
 
rhs 
ω
​
 
0
​
  
​
 
其中：
\(A_{uu}\)：(1/dt)I + 0.5·ω₁×（质量矩阵 + 半隐式对流项）
\(A_{u\omega}\)：(0.5/Re)·∇×（涡量的旋度项，耦合到速度方程）
\(A_{up}\)：∇（压力梯度算子）
\(A_{\omega u}\)：∇×（速度到涡量的旋度耦合）
\(A_{\omega\omega}\)：-I（涡量恒等约束 
∇
×
u
1
−
ω
2
=
0
∇×u 
1
​
 −ω 
2
​
 =0）
\(A_{pu}\)：散度约束 
∇
⋅
u
1
=
0
∇⋅u 
1
​
 =0
这是一个典型的广义鞍点问题，右下角 
3
×
3
3×3 块结构中有零块。
三、线性求解器的三层嵌套 FieldSplit 策略
代码使用了 PETSc 的 PCFieldSplit 预条件子，采用三层嵌套 Schur 补分裂。
第一层（外层）：u/p 分裂
solve_linear_system_basic 中（第 1256-1274 行），将系统分为两个场：
u 场：速度 + 涡量（6 个分量：3 个速度 + 3 个涡量）
p 场：压力（1 个分量）
linearsolver.cpp
Lines 1262-1274
  {    IS isU = NULL, isP = NULL;    PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));    // 使用小写 split 名称，命令行前缀对应为 fieldsplit_u / fieldsplit_p    if (isU)      PetscCall(PCFieldSplitSetIS(pc, "u", isU));    if (isP)      PetscCall(PCFieldSplitSetIS(pc, "p", isP));    // ...  }
索引集的构建由 build_up_fieldsplit_is 完成（第 72-159 行），根据 DM 的 DOF 布局区分：
two-form 系统（dof3 > 0, dof0 == 0）：u₂ 在面（LEFT/DOWN/BACK）、ω₁ 在棱（BACK_DOWN/BACK_LEFT/DOWN_LEFT）、P₃ 在单元中心（ELEMENT）
one-form 系统：u₁ 在棱、ω₂ 在面、P₀ 在顶点（BACK_DOWN_LEFT）
使用上三角 Schur 补分解：
linearsolver.cpp
Lines 806-807
  PetscCall(PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR));  PetscCall(PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_UPPER));
数学上等价于：
[
A
u
B
C
0
]
=
[
I
0
C
A
u
−
1
I
]
[
A
u
B
0
S
p
]
[ 
A 
u
​
 
C
​
  
B
0
​
 ]=[ 
I
CA 
u
−1
​
 
​
  
0
I
​
 ][ 
A 
u
​
 
0
​
  
B
S 
p
​
 
​
 ]
其中 
S
p
=
−
C
A
u
−
1
B
S 
p
​
 =−CA 
u
−1
​
 B 是压力 Schur 补。
压力 Schur 补的近似预条件：
two-form 系统：
S
p
≈
−
1
γ
M
p
S 
p
​
 ≈− 
γ
1
​
 M 
p
​
 （build_pressure_schur_user_mat，第 481-516 行），其中 
γ
γ 是 grad-div 稳定化参数，
M
p
M 
p
​
  是压力质量矩阵（lumped 对角）。
one-form / half 系统：
S
p
≈
−
1
α
+
1
/
Δ
t
I
S 
p
​
 ≈− 
α+1/Δt
1
​
 I（build_oneform_pressure_identity_schur_user_mat，第 518-553 行），利用了速度块中 
1
Δ
t
I
Δt
1
​
 I 占主导的结构。
第二层（中层）：u 块内部的 v/w 分裂
u 场（速度 + 涡量）内部进一步被分裂为：
v 场：纯速度自由度
w 场：纯涡量自由度
linearsolver.cpp
Lines 897-904
    if (canNestedVW) {      PetscCall(PCSetType(subpc, PCFIELDSPLIT));      PetscCall(PCFieldSplitSetDetectSaddlePoint(subpc, PETSC_FALSE));      PetscCall(PCFieldSplitSetType(subpc, PC_COMPOSITE_SCHUR));      PetscCall(          PCFieldSplitSetSchurFactType(subpc, PC_FIELDSPLIT_SCHUR_FACT_UPPER));      PetscCall(PCFieldSplitSetIS(subpc, "w", isWSub));      PetscCall(PCFieldSplitSetIS(subpc, "v", isVSub));
u 块的矩阵结构为：
A
u
=
[
A
v
v
A
v
w
A
w
v
A
w
w
]
A 
u
​
 =[ 
A 
vv
​
 
A 
wv
​
 
​
  
A 
vw
​
 
A 
ww
​
 
​
 ]
关键性质：由于涡量约束 
∇
×
u
−
ω
=
0
∇×u−ω=0 给出 
A
w
w
=
−
I
A 
ww
​
 =−I，内层的速度 Schur 补可以精确计算：
S
v
=
A
v
v
−
A
v
w
A
w
w
−
1
A
w
v
=
A
v
v
+
A
v
w
⋅
A
w
v
S 
v
​
 =A 
vv
​
 −A 
vw
​
 A 
ww
−1
​
 A 
wv
​
 =A 
vv
​
 +A 
vw
​
 ⋅A 
wv
​
 
这在 build_velocity_schur_precond_mat（第 561-596 行）中实现：
linearsolver.cpp
Lines 577-585
  Mat Avv = NULL, Avw = NULL, Awv = NULL, CurlCurl = NULL;  PetscCall(MatCreateSubMatrix(A, isV, isV, MAT_INITIAL_MATRIX, &Avv));  PetscCall(MatCreateSubMatrix(A, isV, isW, MAT_INITIAL_MATRIX, &Avw));  PetscCall(MatCreateSubMatrix(A, isW, isV, MAT_INITIAL_MATRIX, &Awv));  PetscCall(MatMatMult(Avw, Awv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CurlCurl));  PetscCall(MatDuplicate(Avv, MAT_COPY_VALUES, Pv));  PetscCall(MatAXPY(*Pv, 1.0, CurlCurl, DIFFERENT_NONZERO_PATTERN));
物理意义：
S
v
S 
v
​
  包含了 \(\frac{1}{\Delta t}I + \text{对流} + \frac{1}{2Re}\nabla \times \nabla \times + \text{grad-div 增强}\)，即一个椭圆型算子。
第三层（最内层）：v 子块的求解器
对于 v 子块（速度 Schur 补）的求解，根据系统类型选择不同的策略：
two-form 系统：Hypre ADS（Auxiliary-space Divergence Solver）
linearsolver.cpp
Lines 965-985
        if (isTwoSystem && isVInner) {          // ...            PetscCall(KSPSetFromOptions(inner[j]));            PetscBool isHypre = PETSC_FALSE;            PetscCall(                PetscObjectTypeCompare((PetscObject)ipc, PCHYPRE, &isHypre));            if (isHypre) {              PetscCall(set_twoform_ads_coordinates(dm, ipc));              if (Gads)                PetscCall(PCHYPRESetDiscreteGradient(ipc, Gads));              if (Cads)                PetscCall(PCHYPRESetDiscreteCurl(ipc, Cads));            }
ADS 是专门求解 H(div) 问题的代数多重网格方法，需要三个辅助信息：
坐标（set_twoform_ads_coordinates，第 256-413 行）：面心坐标
离散梯度 G_ads（build_twoform_ads_aux_mats，第 601-793 行）：从单元中心（压力）到面（速度）的拓扑梯度
离散旋度 C_ads：从棱（涡量）到面（速度）的拓扑旋度
这些纯拓扑算子精确编码了 de Rham 复形中的外微分关系：
V
h
0
→
∇
h
V
h
1
→
∇
h
×
V
h
2
→
∇
h
⋅
V
h
3
V 
h
0
​
  
∇ 
h
​
 
​
 V 
h
1
​
  
∇ 
h
​
 ×
​
 V 
h
2
​
  
∇ 
h
​
 ⋅
​
 V 
h
3
​
 
one-form 系统：GAMG（代数多重网格）
对应 quick_test 中的配置：
quick_test
Lines 26-26
  -one_fieldsplit_u_fieldsplit_v_ksp_type fgmres -one_fieldsplit_u_fieldsplit_v_pc_type gamg \
四、Grad-div 稳定化
one-form 系统
在 solve_linear_system_basic（第 1238-1249 行）中添加：
A
eff
=
A
+
α
⋅
(
−
h
2
6
)
⋅
∇
(
∇
⋅
)
A 
eff
​
 =A+α⋅(− 
6
h 
2
 
​
 )⋅∇(∇⋅)
linearsolver.cpp
Lines 1246-1249
    const PetscReal oneCoeff = alphaOneForm * (-(hx * hx) / 6.0);    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Aowned));    Aop = Aowned;    PetscCall(add_graddiv_term_to_matrix(Aop, dm, oneCoeff));
这个 
−
α
h
2
/
6
−αh 
2
 /6 系数来源于论文中 pressure-robust 的分析需要。
two-form 系统
通过 solve_linear_system_graddiv（第 1507-1530 行）添加 
γ
⋅
∇
(
∇
⋅
)
γ⋅∇(∇⋅) 项：
linearsolver.cpp
Lines 1515-1516
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Aeff));  PetscCall(add_graddiv_term_to_matrix(Aeff, dm, gamma));
Grad-div 增强的实现在 add_graddiv_term_to_matrix（第 1080-1163 行）中：提取系统矩阵中的 G = A(u, p)（梯度块）和 D = A(p, u)（散度块），然后将 
γ
⋅
G
⋅
D
γ⋅G⋅D 加到 A(u, u) 子块。这是因为 
∇
(
∇
⋅
)
=
G
⋅
D
∇(∇⋅)=G⋅D。
五、压力处理
零空间处理
当压力不被钉住时（attachPressureNullspace = true），attach_pressure_nullspace_if_needed（第 1019-1075 行）构造一个常压力零空间（仅压力分量为常数 1，速度/涡量分量为 0），附加到矩阵上，并从 RHS 中移除零空间分量。这保证奇异系统的相容性。
压力均值归零
求解完成后（第 1400-1498 行），计算所有压力自由度的全局均值，为后续使用做准备（目前均值减除的代码已被注释掉）。
六、完整的求解流程总结
每个时间步的调用链为：
time_evolve2 (整数步, two-form)  → assemble_2form_system_matrix  (组装矩阵 A 和 rhs)  → solve_linear_system_with_switch("two_", ...)    → solve_linear_system_graddiv  (当 use_graddiv=true)      → add_graddiv_term_to_matrix  (添加 γ·∇(∇·) 到 u-u 块)      → solve_linear_system_basic("two_", ...)        → KSP(FGMRES) + PC(FieldSplit u/p)          ├─ p 块: FGMRES + Jacobi          └─ u 块: PC(FieldSplit v/w)              ├─ w(涡量)块: FGMRES              └─ v(速度)块: FGMRES + Hypre ADS                            (注入 G_ads, C_ads, 坐标)time_evolve1 (半整数步, one-form)  → assemble_1form_system_matrix  (组装矩阵 A 和 rhs)  → solve_linear_system_with_switch("one_", ...)    → solve_linear_system_basic("one_", ...)      → 添加 α·(-h²/6)·∇(∇·) 稳定化到 u-u 块      → KSP(FGMRES) + PC(FieldSplit u/p)        ├─ p 块: FGMRES + Jacobi        └─ u 块: PC(FieldSplit v/w)            ├─ w(涡量)块: FGMRES            └─ v(速度)块: FGMRES + GAMG
初始化的 half 步（compute_half_solution）使用前缀 half_，结构类似但 u 块没有 v/w 内层分裂（因为 half 步只有速度 + 压力，无独立涡量场），回退到单块 ASM/Jacobi 预条件。
七、设计亮点
精确利用了 \(A_{ww} = -I\) 的结构：涡量约束使得内层 Schur 补可以精确计算，而不需要近似。
ADS 与 de Rham 复形的天然契合：two-form 的速度场位于 H(div) 空间，ADS 正是为此类问题设计的求解器，而代码提供的拓扑梯度/旋度矩阵正好编码了离散 de Rham 复形的结构。
Pressure-robust：论文证明了误差估计中压力项 
A
1
,
4
=
0
A 
1,4
​
 =0（公式 4.13），在代码中对应 Schur 补分裂后压力块不污染速度精度。
命令行可配置：所有求解器参数都可通过 PETSc 命令行选项覆盖，便于调试和优化。
利用schur补求解u块的时候，是先求解的w还是v?

根据代码中的注册顺序和 Schur 分解类型，先求解 v（速度），再回代求解 w（涡量）。
具体逻辑如下：
注册顺序
linearsolver.cpp
Lines 903-904
      PetscCall(PCFieldSplitSetIS(subpc, "w", isWSub));      PetscCall(PCFieldSplitSetIS(subpc, "v", isVSub));
w 先注册（field 0），v 后注册（field 1）。
UPPER Schur 分解的求解顺序
linearsolver.cpp
Lines 901-902
      PetscCall(PCFieldSplitSetType(subpc, PC_COMPOSITE_SCHUR));      PetscCall(          PCFieldSplitSetSchurFactType(subpc, PC_FIELDSPLIT_SCHUR_FACT_UPPER));
对于系统：
[
A
w
w
A
w
v
A
v
w
A
v
v
]
[
x
w
x
v
]
=
[
b
w
b
v
]
[ 
A 
ww
​
 
A 
vw
​
 
​
  
A 
wv
​
 
A 
vv
​
 
​
 ][ 
x 
w
​
 
x 
v
​
 
​
 ]=[ 
b 
w
​
 
b 
v
​
 
​
 ]
PETSc 的 UPPER 分解始终对 field 1（即后注册的 v）构造 Schur 补：
S
v
=
A
v
v
−
A
v
w
A
w
w
−
1
A
w
v
S 
v
​
 =A 
vv
​
 −A 
vw
​
 A 
ww
−1
​
 A 
wv
​
 
求解步骤为：
先解 v：
S
v
⋅
x
v
=
b
v
−
A
v
w
A
w
w
−
1
b
w
S 
v
​
 ⋅x 
v
​
 =b 
v
​
 −A 
vw
​
 A 
ww
−1
​
 b 
w
​
 
再回代解 w：
A
w
w
⋅
x
w
=
b
w
−
A
w
v
⋅
x
v
A 
ww
​
 ⋅x 
w
​
 =b 
w
​
 −A 
wv
​
 ⋅x 
v
​
 
代码中的注释也明确写了这一点：
linearsolver.cpp
Lines 931-932
        // 硬绑定优先：内层 2 个子块时，j==0 视为 w，j==1 视为 v        // （注册顺序为 w 先、v 后，UPPER Schur 先解 field1=v 再回代 field0=w）
为什么这样设计
因为 
A
w
w
=
−
I
A 
ww
​
 =−I（涡量约束 
∇
×
u
−
ω
=
0
∇×u−ω=0），所以：
A
w
w
−
1
=
−
I
A 
ww
−1
​
 =−I，求逆是 trivial 的
Schur 补 
S
v
=
A
v
v
+
A
v
w
A
w
v
S 
v
​
 =A 
vv
​
 +A 
vw
​
 A 
wv
​
  是一个椭圆型算子，物理上对应 
1
Δ
t
I
+
对流
+
1
2
R
e
∇
×
∇
×
Δt
1
​
 I+对流+ 
2Re
1
​
 ∇×∇×，适合用 ADS/GAMG 求解
回代步 
x
w
=
−
(
b
w
−
A
w
v
x
v
)
x 
w
​
 =−(b 
w
​
 −A 
wv
​
 x 
v
​
 ) 只需一次矩阵向量乘和取反，无需迭代
所以这个顺序的设计完全利用了 
A
w
w
=
−
I
A 
ww
​
 =−I 的结构优势——把计算量集中在 v 的 Schur 补上，w 的回代几乎没有代价。
对于half步，这个线性求解器代码是如何求解的？

现在完全理清了 half 步的求解路径。下面详细解释。
Half 步的特殊性：只有速度 + 压力，没有涡量
1. Half 步的物理背景
Half 步对应论文 Section 5 的 Starting procedure（第 12 页）。为启动 leapfrog 交错时间步进，需要从整数时刻 
t
=
0
t=0 的初值 
(
u
1
h
,
0
,
ω
1
h
,
0
,
ω
2
h
,
0
)
(u 
1
h,0
​
 ,ω 
1
h,0
​
 ,ω 
2
h,0
​
 )，用显式欧拉推算出半整数时刻 
t
=
Δ
t
/
2
t=Δt/2 的 
u
1
h
,
1
/
2
u 
1
h,1/2
​
  和 
ω
2
h
,
1
/
2
ω 
2
h,1/2
​
 。离散方程为：
u
1
h
,
1
/
2
Δ
t
+
∇
P
0
h
,
0
=
R
1
f
0
−
ω
1
h
,
0
×
u
1
h
,
0
−
1
R
e
∇
×
ω
2
h
,
0
+
u
1
h
,
0
Δ
t
Δt
u 
1
h,1/2
​
 
​
 +∇P 
0
h,0
​
 =R 
1
​
 f 
0
 −ω 
1
h,0
​
 ×u 
1
h,0
​
 − 
Re
1
​
 ∇×ω 
2
h,0
​
 + 
Δt
u 
1
h,0
​
 
​
 
∇
⋅
u
1
h
,
1
/
2
=
0
∇⋅u 
1
h,1/2
​
 =0
注意：涡量和对流项全部被显式处理到右端项，左端矩阵中只剩 
u
1
u 
1
​
  和 
P
0
P 
0
​
 。
2. 创建专用 DM：无涡量自由度
time_evolve.cpp
Lines 542-543
    DM dmHalf = NULL;    PetscCall(DMStagCreateCompatibleDMStag(dmSol_1, 1, 1, 0, 0, &dmHalf));
参数 (1, 1, 0, 0) 意味着：
dof0 = 1：顶点上有 1 个自由度 → 压力 
P
0
P 
0
​
 
dof1 = 1：棱上有 1 个自由度 → 速度 
u
1
u 
1
​
 
dof2 = 0：面上没有自由度 → 没有涡量 \(\omega_2\)
dof3 = 0：单元中心没有自由度
所以 half 步的线性系统只有两个未知量场：速度 \(u_1\)（棱）和压力 \(P_0\)（顶点）。
3. 矩阵结构：经典 Stokes 型鞍点系统
矩阵只组装了三个算子：
time_evolve.cpp
Lines 557-571
    // 2.1 时间导数矩阵：1/dt * I（对 u₁）    PetscCall(assemble_u1_dt_matrix(dmHalf, A, this->dt));    // 2.2 压力梯度矩阵：∇（对 P₀）    PetscCall(assemble_p0_gradient_matrix(dmHalf, A));    // 2.3 散度矩阵：∇·（对 u₁）    PetscCall(assemble_u1_divergence_matrix(dmHalf, A));
系统矩阵的结构为：
[
1
Δ
t
I
∇
∇
⋅
0
]
[
u
1
P
0
]
=
[
u
1
h
,
0
Δ
t
+
R
1
f
0
−
ω
1
0
×
u
1
0
−
1
R
e
∇
×
ω
2
0
0
]
[ 
Δt
1
​
 I
∇⋅
​
  
∇
0
​
 ][ 
u 
1
​
 
P 
0
​
 
​
 ]=[ 
Δt
u 
1
h,0
​
 
​
 +R 
1
​
 f 
0
 −ω 
1
0
​
 ×u 
1
0
​
 − 
Re
1
​
 ∇×ω 
2
0
​
 
0
​
 ]
这是一个标准的 Stokes 型鞍点问题，比 one/two 步简单得多——没有涡量耦合块。
4. 求解器在 linearsolver.cpp 中的路径
调用入口：
time_evolve.cpp
Lines 630-630
    PetscCall(solve_linear_system_with_switch(A, rhs, sol_half, dmHalf, "half_", ...));
进入 solve_linear_system_basic 后，关键的判断链如下：
(a) Alpha 稳定化
isHalfSystem = true，所以进入稳定化分支：
linearsolver.cpp
Lines 1238-1249
  if (isOneSystem || isHalfSystem) {    // ...    const PetscReal oneCoeff = alphaOneForm * (-(hx * hx) / 6.0);    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Aowned));    Aop = Aowned;    PetscCall(add_graddiv_term_to_matrix(Aop, dm, oneCoeff));  }
在 
1
Δ
t
I
Δt
1
​
 I 的基础上，向速度块添加 
α
⋅
(
−
h
2
/
6
)
⋅
∇
(
∇
⋅
)
α⋅(−h 
2
 /6)⋅∇(∇⋅) 项，使系统变为：
A
u
u
eff
=
1
Δ
t
I
+
α
⋅
(
−
h
2
6
)
∇
(
∇
⋅
)
A 
uu
eff
​
 = 
Δt
1
​
 I+α⋅(− 
6
h 
2
 
​
 )∇(∇⋅)
(b) 外层 u/p FieldSplit
build_up_fieldsplit_is 对 half 的 DM（dof0=1, dof1=1, dof2=0, dof3=0）构建索引：
linearsolver.cpp
Lines 119-144
          // one-form/half: u1 在棱(dof1)，omega2 在面(dof2, half 步可能无)，p 在顶点(dof0)          if (dof1 > 0) {  // ✓ dof1=1, 速度 u1 进入 u 场            s.loc = BACK_DOWN; stU.push_back(s);            s.loc = BACK_LEFT; stU.push_back(s);            s.loc = DOWN_LEFT; stU.push_back(s);          }          if (dof2 > 0) {  // ✗ dof2=0, 没有涡量进入 u 场            // ...          }          if (dof0 > 0) {  // ✓ dof0=1, 压力 P0 进入 p 场            s.loc = BACK_DOWN_LEFT; stP.push_back(s);          }
所以外层分裂为：
u 场：仅速度 
u
1
u 
1
​
 （棱上 3 个方向），不含涡量
p 场：压力 
P
0
P 
0
​
 （顶点）
同样使用上三角 Schur 补分解，压力 Schur 补预条件为：
linearsolver.cpp
Lines 829-837
  } else if ((isOneSystem || isHalfSystem) && dt > 0.0) {    Mat Suser = NULL;    PetscCall(build_oneform_pressure_identity_schur_user_mat(        A, dm, alphaOneForm, dt, &Suser));    // S_p ≈ -1/(α + 1/dt) * I
即 
S
p
≈
−
1
α
+
1
/
Δ
t
I
S 
p
​
 ≈− 
α+1/Δt
1
​
 I，这是对真实 Schur 补 
−
∇
⋅
(
1
Δ
t
I
)
−
1
∇
≈
−
Δ
t
⋅
Δ
−∇⋅( 
Δt
1
​
 I) 
−1
 ∇≈−Δt⋅Δ 的对角近似。
(c) u 块不做内层 v/w 分裂——直接回退到单块求解
这是 half 步与 one/two 步的最大区别。在 configure_fieldsplit_subksp_defaults 中：
linearsolver.cpp
Lines 887-895
    IS isVSub = NULL, isWSub = NULL;    PetscInt nVSub = 0, nWSub = 0;    PetscCall(build_vw_subspace_is(dm, &isVSub, &isWSub));    // ...    const PetscBool canNestedVW =        (PetscBool)(isVSub && isWSub && nVSub > 0 && nWSub > 0);
由于 dof2 = 0，build_vwp_fieldsplit_is 中 stW 为空（没有面自由度做涡量），isWSub = NULL，所以 canNestedVW = false。
进入回退分支：
linearsolver.cpp
Lines 997-1003
    } else {      // 保护逻辑：无双场可分时，直接回退到单块预条件。      PetscCall(PCSetType(subpc, PCASM));      PetscCall(KSPSetFromOptions(subksp[i]));    }
代码默认给 u 块一个 PCASM（加性 Schwarz），但随后 KSPSetFromOptions 会读取命令行参数进行覆盖。
(d) 命令行实际配置
quick_test 中对 half 的配置：
quick_test
Lines 67-75
  -half_ksp_type fgmres -half_pc_type fieldsplit \  -half_pc_fieldsplit_type schur -half_pc_fieldsplit_schur_fact_type upper \  -half_fieldsplit_p_ksp_type fgmres -half_fieldsplit_p_pc_type gamg \  -half_fieldsplit_p_ksp_rtol 1e-2 \  -half_fieldsplit_u_ksp_type fgmres -half_fieldsplit_u_pc_type jacobi \  -half_fieldsplit_u_ksp_atol 1e-5 \  -half_ksp_rtol 1e-5 -half_ksp_atol 1e-5 -half_ksp_max_it 500 \  -half_ksp_monitor_true_residual -half_ksp_converged_reason -half_ksp_view -half_pc_view \  -half_ksp_error_if_not_converged \
实际覆盖后，u 块使用的是 FGMRES + Jacobi（而非代码默认的 ASM），p 块用 FGMRES + GAMG。
5. 总结：Half 步的完整求解流程
compute_half_solution  │  ├─ 创建 dmHalf (dof0=1, dof1=1, dof2=0, dof3=0)  ← 只有 u1 + P0  ├─ 组装矩阵：(1/dt)I + ∇ + ∇·                     ← 标准 Stokes 鞍点  ├─ 组装 RHS：u1_0/dt + f - ω1×u1 - (1/Re)∇×ω2     ← 涡量全在 RHS  │  └─ solve_linear_system_basic("half_", ...)       │       ├─ 添加 α·(-h²/6)·∇(∇·) 稳定化到 u-u 块       │       ├─ 外层 FieldSplit: u(仅速度) / p(压力)       │   ├─ Schur 补分解 (UPPER)       │   ├─ S_p ≈ -1/(α+1/dt) · I  (对角 Schur 预条件)       │   │       │   ├─ p 块: FGMRES + GAMG      ← 命令行覆盖       │   └─ u 块: FGMRES + Jacobi    ← 命令行覆盖（无内层 v/w 分裂！）       │          ↑       │       canNestedVW = false (因为 dof2=0, 没有涡量场)       │       代码默认 PCASM，被命令行覆盖为 Jacobi       │       └─ 求解后提取 u1_half，再算 ω2_half = ∇×u1_half
核心区别：Half 步是一个简单的二场（u, p）Stokes 鞍点问题，只有一层 FieldSplit（u/p），不存在内层的 v/w 嵌套分裂。涡量 
ω
2
h
,
1
/
2
ω 
2
h,1/2
​
  不是通过求解得到的，而是求解后显式计算 ∇× u1_half 得到。

  如果用 PETSc 风格（半时间步 + 同一方程结构）
核心思路：两个场都从 $t=0$ 的同一初始条件出发，第一个半步用 $\Delta t/2$ 作为时间步长，求解与正常半步完全相同结构的方程 (5.4)-(5.6)。
具体来说，在 1/2 步中求解的方程是：
$$\boxed{\frac{u_1^{h,1/2} - u_1^{h,0}}{\Delta t/2} + \omega_1^{h,0} \times \frac{u_1^{h,1/2} + u_1^{h,0}}{2} + \frac{1}{Re} \nabla_h \times \frac{\omega_2^{h,1/2} + \omega_2^{h,0}}{2} + \nabla_h P_0^{h,0} = R_1 f^0}$$
$$\boxed{\nabla_h \times u_1^{h,1/2} - \omega_2^{h,1/2} = 0}$$
$$\boxed{\nabla_h \cdot u_1^{h,1/2} = 0}$$
其中所有 $t=0$ 时刻的量都来自初始条件：
$u_1^{h,0}$, $\omega_2^{h,0}$：半整数场在 $t=0$ 的初始值
$\omega_1^{h,0}$：整数场在 $t=0$ 的涡度（已知）