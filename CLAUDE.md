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

## Stage Transition Mechanism (Training Workflow)

P2PFL uses a state machine architecture where each training round goes through multiple stages. Understanding this mechanism is crucial for debugging and extending the framework.

### Complete Stage Flow

```
StartLearningStage → VoteTrainSetStage → {TrainStage | WaitAggregatedModelsStage} → GossipModelStage → RoundFinishedStage → [Next Round]
```

### Stage Details and Transition Conditions

#### 1. StartLearningStage
**File**: `p2pfl/stages/base_node/start_learning_stage.py`
- **Purpose**: Initialize learning round, wait for network convergence
- **Key Actions**:
  - Wait for heartbeat convergence (`Settings.heartbeat.WAIT_CONVERGENCE` seconds)
  - Gossip initial model to neighbors
  - Prepare for vote phase
- **Transition Condition**: Always proceeds after convergence wait
- **Next Stage**: `VoteTrainSetStage`
- **Blocking**: Yes - sleeps for heartbeat convergence time

#### 2. VoteTrainSetStage  
**File**: `p2pfl/stages/base_node/vote_train_set_stage.py`
- **Purpose**: Select which nodes will participate in training this round
- **Key Actions**:
  - Each node votes for training candidates with weights
  - Broadcast votes using gossip protocol
  - Aggregate votes to determine final training set
- **Transition Condition**: **BRANCHING LOGIC**
  - If `state.addr in state.train_set` → `TrainStage` (selected for training)
  - If not selected → `WaitAggregatedModelsStage` (observer mode)
- **Blocking**: Yes - 60-second timeout waiting for votes
- **Critical Code**:
  ```python
  # vote_train_set_stage.py:109-126
  timeout_occurred = communication_protocol.wait_for_messages(
      60, lambda: len(state.train_set_votes) >= expected_votes
  )
  ```

#### 3A. TrainStage (Training Nodes Path)
**File**: `p2pfl/stages/base_node/train_stage.py`
- **Purpose**: Perform local training and model aggregation
- **Key Actions**:
  - Local model training (`learner.fit()`)
  - Save model weights to disk and generate torrents
  - Add local model to aggregator
  - **CRITICAL BLOCKING**: Wait for aggregation completion
  - Broadcast aggregation status
- **Transition Condition**: Training and aggregation completed
- **Next Stage**: `GossipModelStage`
- **Blocking**: Yes - `aggregator.wait_and_get_aggregation()` blocks until all training nodes contribute
- **Critical Code**:
  ```python
  # train_stage.py:105
  agg_model = aggregator.wait_and_get_aggregation()  # BLOCKS HERE
  ```

#### 3B. WaitAggregatedModelsStage (Non-Training Nodes Path)
**File**: `p2pfl/stages/base_node/wait_agg_models_stage.py`  
- **Purpose**: Wait for training nodes to complete aggregation
- **Key Actions**:
  - Wait for aggregation event from training nodes
  - Handle timeout scenarios gracefully
  - Broadcast readiness signal
- **Transition Condition**: Receives aggregation completion event OR timeout
- **Next Stage**: `GossipModelStage`
- **Blocking**: Yes - event-based waiting with timeout protection
- **Critical Code**:
  ```python
  # wait_agg_models_stage.py:50
  event_set = state.aggregated_model_event.wait(timeout=Settings.training.AGGREGATION_TIMEOUT)
  ```

#### 4. GossipModelStage
**File**: `p2pfl/stages/base_node/gossip_model_stage.py`
- **Purpose**: Distribute aggregated model to all network nodes
- **Key Actions**:
  - Gossip final aggregated model using P2P protocol
  - Ensure all nodes receive the updated model
  - Handle network failures and retransmissions
- **Transition Condition**: Gossip protocol completion
- **Next Stage**: `RoundFinishedStage`
- **Blocking**: Yes - gossip protocol runs until convergence or timeout

#### 5. RoundFinishedStage
**File**: `p2pfl/stages/base_node/round_finished_stage.py`
- **Purpose**: Clean up round state and decide next action
- **Key Actions**:
  - Clear aggregator state
  - Increment round counter
  - Evaluate final metrics (if last round)
- **Transition Condition**: Round count check
  - If `state.round < state.total_rounds` → `VoteTrainSetStage` (next round)
  - Else → `None` (training complete)
- **Blocking**: No - immediate transition

### Synchronization Points and Blocking Mechanisms

#### Vote Synchronization
- **Location**: VoteTrainSetStage
- **Mechanism**: 60-second timeout waiting for all neighbor votes
- **Fallback**: Proceeds with available votes on timeout
- **Impact**: Can delay entire round if nodes are slow to vote

#### Aggregation Synchronization  
- **Location**: TrainStage (training nodes) + WaitAggregatedModelsStage (observers)
- **Mechanism**: Event-based blocking with timeout protection
- **Critical Point**: ALL training nodes must contribute before aggregation completes
- **Timeout**: `Settings.training.AGGREGATION_TIMEOUT` (configurable)
- **Impact**: Single slow/failed training node can delay entire network

#### Gossip Synchronization
- **Location**: GossipModelStage  
- **Mechanism**: Asynchronous gossip with convergence detection
- **Resilience**: Can handle node failures and network partitions
- **Impact**: Generally robust but can be slow on poor networks

### Performance Considerations

1. **Bottlenecks**: 
   - Vote timeout (60s fixed)
   - Aggregation waiting (depends on slowest training node)
   - Gossip convergence (network-dependent)

2. **Failure Modes**:
   - Training node failure → aggregation timeout → round fails
   - Network partition → gossip incomplete → model inconsistency
   - Vote timeout → reduced training set → performance impact

3. **Scalability Limits**:
   - More training nodes = longer aggregation wait
   - Larger networks = slower gossip convergence
   - Vote collection scales with network size

### Debugging Stage Transitions

**Key Log Messages to Monitor**:
```bash
# Stage transitions
"🏃 Running stage: StageName"

# Vote phase
"🗳️ Sending train set vote"
"🚂 Train set of X nodes"

# Training phase  
"🏋️‍♀️ Training..."
"🎓 Training done"

# Aggregation
"⏳ Waiting aggregation"
"✅ Aggregation event received"

# Gossip
"🗣️ Gossiping aggregated model"

# Round completion
"🎉 Round X of Y finished"
```

**Common Issues**:
- Stuck in vote phase → Check network connectivity and timeout settings
- Aggregation timeout → Check if all training nodes are responsive
- Gossip delays → Monitor network conditions and gossip parameters

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