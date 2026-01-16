import pennylane as qml
import jax
import jax.numpy as jnp
from pennylane import numpy as np
import optax
import math 

class QuantumActor:
    def __init__(self, n_qubits, m_layers, target_dim=5): # <--- 优化: target_dim 作为参数
        self.n_qubits = n_qubits
        self.m_layers = m_layers
        self.target_dim = target_dim
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        # Initialize theta
        key = jax.random.PRNGKey(0)
        self.theta = jax.random.normal(key, shape=(m_layers, n_qubits))

        @qml.qnode(self.dev, interface="jax", diff_method="parameter-shift")
        def circuit(x, theta):
            # --- Key Step 1: Dynamically calculate layers ---
            n_data_layers = math.ceil(len(x) / n_qubits)
            loops = max(n_data_layers, theta.shape[0]) 
            
            # --- Initialization ---
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # --- Dynamic Loop (Data Re-uploading) ---
            for l in range(loops):
                # A. Encoding
                for i in range(n_qubits):
                    idx = l * n_qubits + i
                    if idx < len(x):
                        qml.RX(x[idx], wires=i)
                    else:
                        qml.RX(0.0, wires=i) # Padding
                
                # B. Entanglement
                if n_qubits > 1:
                    for i in range(n_qubits - 1):
                        qml.CZ(wires=[i, i+1])
                
                # C. Variational
                if l < theta.shape[0]:
                    for i in range(n_qubits):
                        qml.RY(theta[l][i], wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # JIT compilation
        self.qnode = jax.jit(circuit)

    def __call__(self, x, theta=None):
        theta = theta if theta is not None else self.theta
        output = self.qnode(x, theta)
        
        res = jnp.stack(output) # [n_qubits]
        
        # --- Output Matching ---
        if self.n_qubits == self.target_dim:
            return res
        elif self.n_qubits > self.target_dim:
            return res[:self.target_dim]
        else:
            # Tiling if qubits < target_dim
            repeats = int(jnp.ceil(self.target_dim / self.n_qubits))
            tiled = jnp.tile(res, repeats)
            return tiled[:self.target_dim]

    def update_params(self, new_theta):
        self.theta = new_theta

    def draw(self, x):
        # qml.draw usually needs the raw function, not the jitted one, 
        # but modern PennyLane handles it well. 
        # Note: expansion_strategy might be needed for intricate loops
        return qml.draw(self.qnode)(x, self.theta)

    def latex(self, x):
        return qml.draw_mpl(self.qnode)(x, self.theta)


class QuantumCritic:
    def __init__(self, n_qubits, m_layers):
        self.n_qubits = n_qubits
        self.m_layers = m_layers
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        key = jax.random.PRNGKey(0)
        self.theta = jax.random.normal(key, shape=(m_layers, n_qubits))

        @qml.qnode(self.dev, interface="jax")
        def circuit(x, theta):
            n_data_layers = math.ceil(len(x) / n_qubits)
            loops = max(n_data_layers, theta.shape[0]) 
            
            # --- Initialization ---
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # --- Dynamic Loop ---
            for l in range(loops):
                # A. Encoding
                for i in range(n_qubits):
                    idx = l * n_qubits + i
                    if idx < len(x):
                        qml.RX(x[idx], wires=i)
                    else:
                        qml.RX(0.0, wires=i)
                
                # B. Entanglement
                if n_qubits > 1:
                    for i in range(n_qubits - 1):
                        qml.CZ(wires=[i, i+1])
                
                # C. Variational
                if l < theta.shape[0]:
                    for i in range(n_qubits):
                        qml.RY(theta[l][i], wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # <--- 修复 3: jax.jit 必须放在 circuit 函数定义的外边 ---
        self.qnode = jax.jit(circuit)

    def __call__(self, x, theta=None):
        theta = theta if theta is not None else self.theta
        # print("Theta Shape Within Critic __call__: ", theta.shape)
        return self.qnode(x, theta)

    def update_params(self, new_theta):
        self.theta = new_theta

    def draw(self, x):
        return qml.draw(self.qnode)(x, self.theta)

    def latex(self, x):
        return qml.draw_mpl(self.qnode)(x, self.theta)

    def decode_op(self, q_values, scale=30, method="mean"):
        """Decode multi-qubit outputs to a single scalar (Value)."""
        # Ensure input is JAX array
        q_array = jnp.stack(q_values) if isinstance(q_values, (list, tuple)) else q_values
        
        if method == "mean":
            # Scale * mean([-1, 1]) -> Value
            return scale * jnp.mean(q_array)
        elif method == "sum":
            return qml.numpy.sum(q_array)
        else:
            raise ValueError("Unknown decoding method")

    def evaluate(self, x, k_shots=10):
        """
        Evaluate the critic value. 
        NOTE: Since we are using analytic simulation (no shots), 
        looping k_shots is redundant. Returns exact expectation.
        """

        return self.qnode(x, self.theta)
