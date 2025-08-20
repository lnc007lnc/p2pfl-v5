# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

P2PFL is a general-purpose open-source library for Decentralized Federated Learning systems using P2P networks and gossip protocols. It supports multiple ML frameworks (PyTorch, TensorFlow/Keras, JAX/Flax) and communication protocols.

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