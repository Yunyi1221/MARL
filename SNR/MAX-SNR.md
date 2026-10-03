# 优化问题:

目标函数为：
$$
\underset{\left\{z_{b, u},a_{b,k,u}\right\}}{\operatorname{maximize}}   \sum_{u \in \mathcal{U}}\sum_{b \in \mathcal{A}}\sum_{k \in \mathcal{K}} \log(1+\gamma_{b, k, u})
$$
其中 $\gamma_{b, k, u}$ 定义为：
$$
\gamma_{b, k, u} =
\frac{
    z_{b, u}
    a_{b, k, u}
    P_{b, k,u} H_{b,k,u}
}{
    \sum_{\substack{
        b^{\prime} \in \mathcal{A} \\
        u^{\prime} \neq u
    }}
    z_{b^{\prime}, u^{\prime}}
    a_{b^{\prime}, k, u^{\prime}}
    P_{b^{\prime},k,u^{\prime}} H_{b^{\prime},k, u^{\prime}} 
    + 
    n_{0}
}
$$
$$
\begin{align}
\text{s.t.} \quad & \sum_{b \in \mathcal{B}} z_{b,u} \leq 1, \quad \forall u \in \mathcal{U} \\
& \sum_{u \in \mathcal{U}} z_{b,u} \leq N_b, \quad \forall b \in \{B\} \\
& \sum_{k \in \mathcal{K}} a_{b,k,u} \leq N_{\text{rb}}, \quad \forall b \in \mathcal{B}, u \in \mathcal{U} \\
& z_{b,u} \in \{0,1\}, \quad a_{b,k,u} \in \{0,1\}
\end{align}
$$

## 启发式MAX-SNR算法

该算法通过降序遍历潜在链路-RB组合，仅在新增关联严格提升系统总速率时才激活关联，满足所有约束。

------

### **Algorithm 4: Heuristic MAX-SNR for Joint User Association and RB Allocation**

**目标:** 最大化 $\sum_{u \in \mathcal{U}} \sum_{b \in \mathcal{A}} \sum_{k \in \mathcal{K}} \log(1 + \gamma_{b,k,u})$
**约束:** 用户单连接 ($\sum_b z_{b,u} \leq 1$)、基站容量 ($\sum_u z_{b,u} \leq N_b$)、RB分配限制 ($\sum_k a_{b,k,u} \leq N_{\text{rb}}$)、二元决策变量。

------

#### **伪代码**

**Initialize:**
  Set $ z_{b,u} = 0,  a_{b,k,u} = 0,  \forall b \in \mathcal{B}, u \in \mathcal{U}, k \in \mathcal{K} $
  Set total rate $ R^0 = 0 $

**Step 1: Generate candidate set and sort by SNR**
  For all possible $(b, u, k)$ triples where $b \in \mathcal{A}_u$:
    Calculate estimated SINR $ \tilde{\gamma}_{b,k,u} = \frac{P_{b,k,u} H_{b,k,u}}{n_0} $  // Ignore interference initially
  Sort all $(b, u, k)$ in descending order of $ \tilde{\gamma}_{b,k,u} $ into sequence $\mathcal{Z} = \{ z_1, \dots, z_P \}$
  Let $\delta$ map each triple to its sorted index: $\delta(b,u,k) = p$ and $z_p = (b,u,k)$

**Step 2: Iterative activation**
  for $p = 1$ to $P$ do:
    Let $(b, u, k) = z_p$
    // Check constraints (if violated, skip activation)
    if $\sum_{b'} z_{b',u} \geq 1$ OR $\sum_{u'} z_{b,u'} \geq N_b$ OR $\sum_{k'} a_{b,k',u} \geq N_{\text{rb}}$ then:
      Skip this triple (continue loop)
    else:
      Tentatively set $a_{b,k,u} = 1$ and $z_{b,u} = 1$  // Activate link-RB pair
      Compute current sum-rate $R^p$ with updated interference:
          $\gamma_{b,k,u} = \frac{ P_{b,k,u} H_{b,k,u} }{ \sum\limits_{\substack{(b',u',k') \in \mathcal{S}_p \\ u' \neq u}} P_{b',k,u'} H_{b',k,u'} + n_0 }$  // $\mathcal{S}_p$: currently active triples
      if $R^p > R^{p-1}$ then:  
          Keep activation: $a_{b,k,u} = 1$, $z_{b,u} = 1$, $R^p$ remains
      else:
          Revert: $a_{b,k,u} = 0$, $z_{b,u} = 0$ (if no other RB active for $(b,u)$)
      end if
    end if
  end for

**Output:**
  Recover active triples via $\delta^{-1}$: Final $z_{b,u}$ and $a_{b,k,u}$

1. **排序依据: 无干扰SNR**
    初始化时用 $\tilde{\gamma}_{b,k,u} = P_{b,k,u} H_{b,k,u} / n_0$（忽略干扰）降序排序，优先选择信道条件最优的链路-RB组合。
2. **增量式速率验证**
    每次循环仅新增一个 $(b,u,k)$ 组合，并动态更新干扰项（$\sum_{u' \neq u}$）。计算当前总速率 $R^p$ 时，考虑所有已激活的同频段干扰源。
3. **约束实时检查**
    在激活前验证：
   - 用户 $u$ 是否已关联其他基站 ($\sum_b z_{b,u} \geq 1$)
   - 基站 $b$ 是否达到用户容量上限 ($\sum_u z_{b,u} \geq N_b$)
   - 用户 $u$ 在基站 $b$ 是否超出RB配额 ($\sum_k a_{b,k,u} \geq N_{\text{rb}}$)
4. **贪婪保留准则**
    仅当新增组合使 $R^p > R^{p-1}$（严格提升）时才保留，否则撤销本次操作。确保每一步迭代均单调提升目标函数。
5. **复杂度控制**
    通过预排序和单链路增量更新，避免全局搜索。计算开销集中在速率更新时干扰项的重新计算（可优化为局部干扰更新）。

------

