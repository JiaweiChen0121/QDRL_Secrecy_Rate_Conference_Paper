# quantum_models.py
import pennylane as qml
import jax
import jax.numpy as jnp
from pennylane import numpy as np
import optax

class QuantumActor:
    def __init__(self, n_qubits, m_layers):
        self.n_qubits = n_qubits
        self.m_layers = m_layers
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        #self.theta = np.random.randn(m_layers, n_qubits, requires_grad=True)
        self.theta = jnp.array(jax.random.normal(jax.random.PRNGKey(0), shape=(m_layers, n_qubits)))

        @qml.qnode(self.dev, interface="jax", diff_method="parameter-shift")
        def circuit(x, theta):
            theta = jnp.asarray(theta)
            #first layer Hadamard layers for initializing superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for l in range(m_layers): # we only have one layers
                for i in range(len(x)): # this is the change we made we don't just go through the number of qubits we have but we go through the whole data
                    qml.RX(x[i], wires=i) # we turn the whole data into quantum rotation angle
                for i in range(n_qubits - 1): #entanglement
                    qml.CZ(wires=[i, i+1])
                for i in range(n_qubits): #trainable parameter
                    qml.RY(theta[l][i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        #self.qnode = circuit
        self.qnode = jax.jit(circuit)

    def __call__(self, x, theta=None):
        theta = theta if theta is not None else self.theta
        # print("Theta Shape Within Actor __call__: ", theta.shape)
        output = self.qnode(x, theta)
        # circuits outputs is  list of scalars (length = n_qubits)
        # Action space is 5-dim (VX, VY, VZ, Power, User_Scheduling)
        
        res = jnp.stack(output) # turn into JAX array [n_qubits]
        
        target_dim = 5 # Action is 5-dim
        
        if self.n_qubits == target_dim:
            return res
        elif self.n_qubits > target_dim:
            return res[:target_dim] # use the first 5
        else:
            # if qubits less than 5 we have to expand it
            tiled = jnp.tile(res, int(jnp.ceil(target_dim / self.n_qubits)))
            return tiled[:target_dim]

    def update_params(self, new_theta):
        self.theta = new_theta

    def draw(self, x):
        return qml.draw(self.qnode)(x, self.theta)

    def latex(self, x):
        return qml.draw_mpl(self.qnode)(x, self.theta)

class QuantumCritic:
    def __init__(self, n_qubits, m_layers):
        self.n_qubits = n_qubits
        self.m_layers = m_layers
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        #self.theta = np.random.randn(m_layers, n_qubits, requires_grad=True)
        self.theta = jnp.array(jax.random.normal(jax.random.PRNGKey(0), shape=(m_layers, n_qubits)))

        @qml.qnode(self.dev, interface="jax", diff_method="parameter-shift")
        def circuit(x, theta):
            theta = jnp.asarray(theta)
            #first layer Hadamard layers for initializing superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for l in range(m_layers): # we only have one layers
                for i in range(len(x)): # this is the change we made we don't just go through the number of qubits we have but we go through the whole data
                    qml.RX(x[i], wires=i) # we turn the whole data into quantum rotation angle
                for i in range(n_qubits - 1): #entanglement
                    qml.CZ(wires=[i, i+1])
                for i in range(n_qubits): #trainable parameter
                    qml.RY(theta[l][i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        #self.qnode = circuit
        self.qnode = jax.jit(circuit)

    def __call__(self, x, theta=None):
        theta = theta if theta is not None else self.theta
        print("Theta Shape Within Critic __call__: ", theta.shape)
        return self.qnode(x, theta)

    def update_params(self, new_theta):
        self.theta = new_theta

    def draw(self, x):
        return qml.draw(self.qnode)(x, self.theta)

    def latex(self, x):
        return qml.draw_mpl(self.qnode)(x, self.theta)

    def decode_op(self, q_values, scale=30, method="mean"):
        """Decode multi-qubit outputs."""
        q_array = jnp.stack(q_values) if isinstance(q_values, (list, tuple)) else q_values
        if method == "mean":
            #return scale * qml.numpy.mean(q_array)
            return scale * jnp.mean(q_array)
        elif method == "sum":
            return qml.numpy.sum(q_array)
        else:
            raise ValueError("Unknown decoding method")

    def evaluate(self, x, k_shots=10):
        total = jnp.zeros(self.n_qubits)
        for k in range(k_shots):
            total += self.qnode(x, self.theta)
        avg_q_per_qubit = total / k_shots
        return avg_q_per_qubit 
