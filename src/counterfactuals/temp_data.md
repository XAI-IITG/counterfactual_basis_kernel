
\section{Methodology}

Let the input time-series for a single engine be a sequence $\mathbf{X} \in \mathbb{R}^{T \times d_{feat}}$, where $T \in \mathbb{N}$ is the sequence length and $d_{feat}$ is the number of sensor features (here, $d_{feat}=14$). In our experiments we have taken $T = 50$. The input instance $X \in \mathbb{R}^{T \times d_{feat}}$ be the query instance(the original time series data) lets say for some specific unit in the case of CMAPSS, for which we seek an explanation for the data be, 

\begin{equation}
	\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T]^\top, \quad \text{where } \mathbf{x}_t \in \mathbb{R}^{d_{feat}}
\end{equation}

We train a predictive model like Transformer or LSTM, let it be 

\begin{equation}
	\label{eq:trained_model}
	f: \mathbb{R}^{T \times d_{feat}} \to \mathbb{R}
\end{equation}

We train the model as defined in Equation \ref{eq:trained_model} to predict the RUL(remaining Useful Life) of a machine/engine  
\begin{equation}
	\label{eq:RUL}
	\hat{y} = f(\mathbf{X}) \in \mathbb{R}^+
\end{equation}

\subsection{Counterfactual Explanations}
Counterfactual explanations answer the question: "What is the minimal change to the input features that would result in a desired different outcome?" In our context, this translates to: "What are the smallest changes in the sensor reading history that would result in the RUL prediction falling within a desired range (e.g., increasing by 10 cycles)?". 
We seek a counterfactual instance $\mathbf{X_{cf}}$ such that $ f(\mathbf{X_{cf}}) \approx y_{target}$, for that we would do a minimal change to the original $\mathbf{X}$ 

\begin{equation}
	\label{eq:cf_basic}
	\mathbf{X_{cf}} = \mathbf{X} + \mathbf{\Delta}
	\quad \text{where } \mathbf{\Delta} \in \mathbb{R}^{T \times d_{feat}}
\end{equation}

From Equation \ref{eq:cf_basic} we define $\mathbf{\Delta}$ as a free parameter matrix, the $\mathbf{\Delta}$ is constrasined to lie in the span of a smooth basis $\mathbf{\Phi}$, $\mathcal{B} = \{ \phi_k \}_{k=1}^K$ be a set of $K$ distinct temporal basis functions, where $K \ll T$. and construct the Basis Projection Matrix $\mathbf{\Phi} \in \mathbb{R}^{T \times K}$

\begin{equation}
	\mathbf{\Phi} = 
	\begin{bmatrix}
		\phi_1(1) & \phi_2(1) & \dots & \phi_K(1) \\
		\phi_1(2) & \phi_2(2) & \dots & \phi_K(2) \\
		\vdots & \vdots & \ddots & \vdots \\
		\phi_1(T) & \phi_2(T) & \dots & \phi_K(T)
	\end{bmatrix}
\end{equation}

We define a learnable weight matrix $\mathbf{W} \in \mathbb{R}^{K \times d_{feat}}$ such that for a $w_{k,m}$ it represents the contribution of the basis function $k$ to sensor $m$. The pertubations $\mathbf{\Delta}(\mathbf{W})$ is defined by the linear projection.

\begin{equation}
	\label{eq:lin_proj}
	\mathbf{\Delta}(\mathbf{W}) = \mathbf{\Phi}\mathbf{W}
\end{equation}

So From Equation \ref{eq:cf_basic} and Equation \ref{eq:lin_proj} the counterfactual parameterized by $\mathbf{W}$ is 

\begin{equation}
	\mathbf{X}_{cf}(\mathbf{W}) = \mathbf{X} + \mathbf{\Phi}\mathbf{W}
\end{equation}

To generate a set of $N$ diverse counterfactuals we optimize on the weights $\mathcal{W} = \{ \mathbf{W}^{(1)}, \dots, \mathbf{W}^{(N)} \}$ such that our total loss is :

\begin{equation}
	\label{eq:cf_total_loss}
	\mathcal{J}(\mathcal{W}) = \lambda_0 \mathcal{L}_{\text{valid}} + \lambda_1 \mathcal{L}_{\text{prox}} + \lambda_2 \mathcal{L}_{\text{sparse}} + \lambda_3 \mathcal{L}_{\text{dpp}} + \lambda_3 \mathcal{L}_{\text{smooth}}
\end{equation}

the \textbf{Validity} Loss $\mathcal{L}_{\text{valid}}$ ensures that the counterfactuals generated achieves the desired prediction 
\begin{equation}
	\label{eq:cf_L_validity}
	\mathcal{L}_{\text{valid}}(\mathcal{W}) = \frac{1}{N} \sum_{i=1}^N \left( f(\mathbf{X} + \mathbf{\Phi}\mathbf{W}^{(i)}) - y_{target} \right)^2
\end{equation}

The \textbf{Proximity} Loss ($\mathcal{L}_{\text{prox}}$) is to penalises large deviations from the original input, encouraging minimal, actionable changes. To achieve this we use our proximity loss similar to the Wachter \cite{wachter2018counterfactualexplanationsopeningblack}. so for each feature d in $d_{feat}$, the $m_{d}^{-1}$ is generally set to $1$ 

\begin{equation}
	med_d = median_j(x_{j, d}),
	\quad MAD_d = median_j(|x_{j,d} -med_d|)
\end{equation}

\begin{equation}
	\label{eq:MAD}
	m_{d}^{-1} = \frac{1}{MAD_d + \epsilon} 
\end{equation}

From Equation \ref{eq:MAD} the Proximity Loss is

\begin{equation}
	\label{eq:L_prox}
	\mathcal{L}_{\text{prox}}(\mathcal{W}) = \frac{1}{N} \sum_{i = 1}^{N} \left\| \mathbf{\Phi}\mathbf{W}^{(i)} m_{d}^{-1} \right\|_F^{2}
\end{equation}

The \textbf{Sparsity} Loss ($\mathcal{L}_{\text{sparse}}$) encourages the explanations that changes only in a small subset of parameters or features. 

\begin{equation}
	\label{eq:cf_L_sparse}
		\mathcal{L}_{\text{sparse}}(\mathcal{W}) = \frac{1}{N} \sum_{i = 1}^{N} \left\| \mathbf{W}^{(i)} \right\|_1
\end{equation}

The \textbf{Smoothness} Loss ($\mathcal{L}_{\text{smooth}}$) regularises temporal irregularities, it penalises rapid, and sudden temporal changes in the counterfactual perturbation in Equation \ref{eq:lin_proj}. by measuring its discrete curvature (second difference) over time. Concretely, for each counterfactual $i$ and feature $d$, $\mathbf{\Delta}^{(i)} = \mathbf{\Phi}\mathbf{W}^{(i)}$, then the second order derivative of $\mathbf{\Delta}$ also known as the Roughness Penalty \cite{green1993nonparametric}. 

\begin{equation}
	\mathbf{\nabla}^2(\mathbf{\Delta}^{(i)})_{t, d} = \mathbf{\Delta}^{(i)}_{t+1, d} -2\mathbf{\Delta}^{(i)}_{t, d} + \mathbf{\Delta}^{(i)}_{t-1, d}
\end{equation}

\begin{equation}
	\mathcal{L}_{\text{smooth}}(\mathcal{W}) = \frac{1}{N(T-2)d_{feat}} \sum_{i=1}^{N}\sum_{t=2}^{T-1}\sum_{2=1}^{d_{feat}} (\mathbf{\Delta}^{(i)}_{t+1, d} -2\mathbf{\Delta}^{(i)}_{t, d} + \mathbf{\Delta}^{(i)}_{t-1, d})^2
\end{equation}

The \textbf{Diversity} Loss via Determinantal Point Processes(DPP) ($\mathcal{L}_{\text{dpp}}$) is to shape diverse trajectories of $N$ counterfactuals. DPP ensures that the set of counterfactuals are non redundant \cite{Kulesza_2012} \cite{DiCE_Mothilal_2020}.

\begin{equation}
	\hat{\mathbf{v}}_i = \frac{\text{vec}(\mathbf{W}^{(i)})}{\left\| \text{vec}(\mathbf{W}^{(i)}) \right\|_{2}} 
\end{equation}

\begin{equation}
	S_{i,j} = {\hat{\mathbf{v}}_i}^T \hat{\mathbf{v}}_i
\end{equation}

\begin{equation}
	\mathcal{L}_{\text{dpp}}(\mathcal{W}) = - log(det(\mathbf{S} + \epsilon \mathbf{I}))
\end{equation}

to find a counterfactual instance $\mathbf{X_{cf}}$ that minimizes a loss function, $\mathcal{J}(\mathcal{W})$, balancing five competing objectives:

\begin{equation}
	\label{eq:cf_opt_obj}
	\begin{aligned}
			\mathbf{X}^* = \arg\min_{X_{cf}} \mathcal{J}(\mathcal{W}) 
		\end{aligned}
\end{equation}


\begin{algorithm}
	\caption{BasisCF Counterfactual Generation}
	\label{alg:basis_cf}
	\begin{algorithmic}[1]
		\REQUIRE Trained Model $f$, Query Instance $\mathbf{X} \in \mathbb{R}^{T \times D}$, Target $y_{target}$, Basis $\mathbf{\Phi} \in \mathbb{R}^{T \times K}$, Number of CFs $N$, Hyperparameters $\lambda$
		\ENSURE Optimal Counterfactuals $\mathbf{X}_{cf}^*$
		
		\STATE \textbf{Initialize} weights $\mathbf{W} \in \mathbb{R}^{N \times K \times D} \sim \mathcal{N}(0, 0.01)$ \COMMENT{Learnable coefficients}
		\STATE Initialize Optimizer (Adam) and Scheduler (CosineAnnealing)
		\STATE $\mathbf{X}_{best} \gets \mathbf{X}$
		\STATE $\mathcal{L}_{best} \gets \infty$
		
		\WHILE{$i < \text{max\_iter}$}
		\STATE \textbf{1. Projection:} Project weights to time domain via basis $\mathbf{\Phi}$
		\STATE \quad $\mathbf{\Delta} \leftarrow \mathbf{\Phi} \cdot \mathbf{W}$ \COMMENT{Shape: $N \times T \times D$}
		
		\STATE \textbf{2. Perturbation:} Apply delta to original input
		\STATE \quad $\mathbf{X}_{cf} \leftarrow \mathbf{X} + \mathbf{\Delta}$
		
		\STATE \textbf{3. Constraints:} Enforce normalized bounds
		\STATE \quad $\mathbf{X}_{cf} \leftarrow \text{Clamp}(\mathbf{X}_{cf}, \text{min}=-3.0, \text{max}=3.0)$
		
		\STATE \textbf{4. Prediction:} Get model output
		\STATE \quad $\hat{\mathbf{y}} \leftarrow f(\mathbf{X}_{cf})$
		
		\STATE \textbf{5. Loss Calculation:}
		\STATE \quad $\mathcal{L}_{valid} \leftarrow \text{MSE}(\hat{\mathbf{y}}, y_{target})$
		\STATE \quad $\mathcal{L}_{prox} \leftarrow ||\mathbf{\Delta} \cdot \text{MAD}^{-1}||_F^2$ \COMMENT{Feature-weighted proximity}
		\STATE \quad $\mathcal{L}_{sparse} \leftarrow ||\mathbf{W}||_1$ \COMMENT{Encourage basis sparsity}
		\STATE \quad $\mathcal{L}_{smooth} \leftarrow ||\nabla^2 \mathbf{\Delta}||^2$ \COMMENT{2nd order temporal smoothness}
		
		\STATE \quad \textbf{if} $N > 1$ \textbf{then}
		\STATE \quad \quad $\mathcal{L}_{div} \leftarrow -\log\det(\mathbf{K}_{\mathbf{W}})$ \COMMENT{DPP Diversity on weights}
		\STATE \quad \textbf{end if}
		
		\STATE \quad $\mathcal{L}_{total} \leftarrow \lambda_{v}\mathcal{L}_{valid} + \lambda_{p}\mathcal{L}_{prox} + \lambda_{s}\mathcal{L}_{sparse} + \lambda_{sm}\mathcal{L}_{smooth} + \lambda_{d}\mathcal{L}_{div}$
		
		\STATE \textbf{6. Update:} 
		\STATE \quad Compute gradients $\nabla_{\mathbf{W}} \mathcal{L}_{total}$
		\STATE \quad Clip gradients: $||\nabla_{\mathbf{W}}|| \le 1.0$
		\STATE \quad $\mathbf{W} \leftarrow \text{Optimizer}(\mathbf{W}, \nabla_{\mathbf{W}})$
		
		\STATE \textbf{7. Tracking:}
		\STATE \quad \textbf{if} $\mathcal{L}_{valid} < \mathcal{L}_{best}$ \textbf{then}
		\STATE \quad \quad $\mathbf{X}_{best} \leftarrow \mathbf{X}_{cf}$
		\STATE \quad \quad $\mathcal{L}_{best} \leftarrow \mathcal{L}_{valid}$
		\STATE \quad \textbf{end if}
		
		\STATE \quad \textbf{if} $|\hat{\mathbf{y}} - y_{target}| < \epsilon$ \textbf{then} \textbf{break} \COMMENT{Early Stopping}
		\ENDWHILE
		
		\STATE \RETURN $\mathbf{X}_{best}$
	\end{algorithmic}
\end{algorithm}
