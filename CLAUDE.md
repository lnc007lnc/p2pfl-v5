# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

P2PFL is a general-purpose open-source library for Decentralized Federated Learning systems using P2P networks and gossip protocols. It supports multiple ML frameworks (PyTorch, TensorFlow/Keras, JAX/Flax) and communication protocols.

## Important: Version Control Guidelines

**ALWAYS commit code changes to Git after modifications to enable easy rollback.**

After making significant code changes:
```bash
# Check status and review changes
git status
git diff

# Add and commit changes
git add .
git commit -m "Description of changes"

# Push to remote repository (if configured)
git push
```

## Development Commands

### Package Management
```bash
# Install dependencies with Poetry (for development)
poetry install -E torch    # Install with PyTorch support
poetry install -E tensorflow  # Install with TensorFlow support  
poetry install -E flax     # Install with JAX/Flax support

# Install for production use
pip install "p2pfl[torch]"
```

### Code Quality and Testing
```bash
# Format code
poetry run ruff format p2pfl

# Check code style
poetry run ruff check p2pfl

# Type checking
poetry run mypy -p p2pfl

# Run tests with coverage
poetry run pytest -v --cov=p2pfl

# Run a specific test file
poetry run pytest test/node_test.py -v
```

### Running Examples
```bash
# Run MNIST example via CLI
python -m p2pfl run p2pfl/examples/mnist/mnist.yaml

# Run nodes directly
python p2pfl/examples/mnist/node1.py --port 6666
python p2pfl/examples/mnist/node2.py --port 6666
```

### Documentation
```bash
# Build documentation
cd docs
make html
```

## Architecture Overview

### Core Components

**Node System (`p2pfl/node.py`, `p2pfl/node_state.py`)**
- Central entity coordinating learning, communication, and aggregation
- Manages node lifecycle, connections, and federated learning rounds
- State machine pattern for managing learning stages

**Communication Layer (`p2pfl/communication/`)**
- Protocol-agnostic design with implementations for gRPC and in-memory protocols
- Command pattern for message passing between nodes
- Gossip protocol support for decentralized communication
- Heartbeat mechanism for node health monitoring

**Learning Framework (`p2pfl/learning/`)**
- Abstraction layer supporting PyTorch Lightning, TensorFlow/Keras, and JAX/Flax
- Model wrapper classes: `LightningModel`, `KerasModel`, `FlaxModel`
- Learner factory pattern for framework-agnostic instantiation
- Dataset management with Hugging Face integration

**Aggregation Strategies (`p2pfl/learning/aggregators/`)**
- Pluggable aggregators: FedAvg, FedMedian, SCAFFOLD, FedProx, Krum, etc.
- Base `Aggregator` class for custom implementations

**Workflow Management (`p2pfl/stages/`)**
- Stage-based learning workflow with customizable stages
- Key stages: `StartLearningStage`, `TrainStage`, `GossipModelStage`, `WaitAggModelsStage`

**Compression Strategies (`p2pfl/learning/compression/`)**
- Model compression for efficient communication
- Strategies: Quantization, Top-K sparsification, Low-Rank Approximation, Differential Privacy

### Key Design Patterns

- **Template Pattern**: Used in `CommunicationProtocol`, `Learner`, `Aggregator` for consistent interfaces
- **Command Pattern**: All network messages inherit from `Command` base class
- **Strategy Pattern**: Aggregation algorithms and compression strategies are interchangeable
- **Factory Pattern**: `LearnerFactory` creates appropriate learner instances based on model type
- **Singleton Pattern**: Used for logging and server management in memory protocol

### Important Files to Understand

- `p2pfl/node.py`: Main node implementation and learning coordination
- `p2pfl/stages/workflows.py`: Defines the federated learning workflow
- `p2pfl/communication/protocols/communication_protocol.py`: Base protocol interface
- `p2pfl/learning/frameworks/learner.py`: Base learner interface
- `p2pfl/learning/aggregators/aggregator.py`: Base aggregator interface

### Testing Approach

The project uses pytest for testing with coverage tracking. Tests are organized by module in the `test/` directory. Key test areas include:
- Node functionality and state management
- Communication protocols and commands
- Learning frameworks and aggregators
- Reproducibility and simulation capabilities

## Message Broadcasting and Communication Mechanisms

### Broadcast Functionality Overview

P2PFL provides a **non-blocking broadcast mechanism** for distributed coordination among nodes. The broadcast functionality is implemented at multiple levels with different reliability guarantees depending on the message type.

#### Basic Broadcast Interface

**Location**: `p2pfl/communication/protocols/communication_protocol.py:116`
```python
def broadcast(self, msg: Any, node_list: Optional[List[str]] = None) -> None
```

**Implementation**: `p2pfl/communication/protocols/protobuff/protobuff_communication_protocol.py:264`
- Sends messages to all directly connected neighbors
- Non-blocking (fire-and-forget) for basic messages
- No acknowledgment required for simple status updates

#### How to Use Broadcasting

```python
# Basic usage pattern in Node class
self._communication_protocol.broadcast(
    self._communication_protocol.build_msg(
        cmd="command_name",      # Command identifier
        args=["arg1", "arg2"],   # Optional arguments
        round=self.round         # Optional round number
    )
)
```

### Message Types and Their Purposes

#### 1. Learning Flow Control
- **StartLearningCommand** (`node.py:371`): Initiates federated learning with parameters (rounds, epochs, experiment_name)
- **StopLearningCommand** (`node.py:391`): Terminates learning process across all nodes

#### 2. Model Synchronization
- **ModelInitializedCommand** (`start_learning_stage.py:76`): Signals model initialization completion
- **ModelsAggregatedCommand** (`train_stage.py:92`): Announces models added to aggregation
- **ModelsReadyCommand** (`train_stage.py:108`): Indicates aggregation completion

#### 3. Voting and Consensus
- **VoteTrainSetCommand** (`vote_train_set_stage.py:103`): Broadcasts node's vote for training participants

#### 4. Metrics Sharing
- **MetricsCommand** (`train_stage.py:227`): Shares evaluation metrics (accuracy, loss, etc.)

### Reliability Mechanisms

#### For Critical Messages (e.g., Voting)

**Not pure fire-and-forget!** Critical messages have additional safeguards:

1. **Timeout Mechanism** (`vote_train_set_stage.py:112-170`)
   - Default timeout: 60 seconds (`Settings.training.VOTE_TIMEOUT`)
   - Continues with available votes after timeout
   - Logs missing votes for debugging

2. **Active Waiting with Locks**
   ```python
   state.wait_votes_ready_lock.acquire(timeout=2)  # Check every 2 seconds
   ```

3. **Post-Validation** (`vote_train_set_stage.py:176-186`)
   - Verifies selected nodes are still online
   - Removes offline nodes from training set

#### For Model Weights (Gossip Protocol)

**Enhanced reliability through Gossip** (`gossiper.py`):

1. **Periodic Retransmission** (`gossiper.py:130-163`)
   - Configurable period and messages per period
   - Continues until acknowledgment or termination

2. **Synchronous Gossip Mode** (`gossiper.py:169-209`)
   - Used for critical model weight transfers
   - Blocks until all target nodes receive data
   - Includes completion detection

### Message Flow Example

```
Node A                     Node B                     Node C
  |                         |                         |
  |--broadcast:StartLearning-->|                      |
  |                         |--broadcast:StartLearning-->|
  |                         |                         |
  |--broadcast:VoteTrainSet--->|                      |
  |<--broadcast:VoteTrainSet---|--broadcast:VoteTrainSet->|
  |                         |<--broadcast:VoteTrainSet---|
  |                         |                         |
  | [Wait up to 60s for all votes]                   |
  | [Process available votes]                         |
  | [Validate and clean results]                      |
```

### Design Philosophy

1. **Fault Tolerance First**: System continues despite partial failures
2. **Eventually Consistent**: Timeouts prevent infinite waiting
3. **Pragmatic Approach**:
   - Simple notifications: Fire-and-forget
   - Critical operations: Additional safeguards
   - Model data: Gossip protocol with retries

### Creating Custom Broadcast Messages

To add a new broadcast message type:

1. Create a new Command class in `p2pfl/communication/commands/message/`
2. Inherit from `Command` base class
3. Implement required methods:
   ```python
   @staticmethod
   def get_name() -> str:
       return "your_command_name"
   
   def execute(self, source: str, round: int, **kwargs) -> Optional[str]:
       # Handle received message
       pass
   ```
4. Register and use in your workflow:
   ```python
   communication_protocol.broadcast(
       communication_protocol.build_msg(YourCommand.get_name(), args)
   )