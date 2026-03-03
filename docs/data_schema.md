# N-BaIoT Dataset Schema

## Dataset Overview
- Total Features: 115
- Classes: 11 (10 attack types + benign)
- Data Type: Numerical tabular
- Device-based data grouping

## Key Considerations for Federated Learning
- Each IoT device can simulate a federated client
- Non-IID partitioning required
- Large dataset (~10GB extracted)
- Memory-efficient loading required

## Planned Pipeline
1. Raw CSV loading
2. Feature normalization
3. Label encoding
4. Client partitioning (IID / Dirichlet α=0.5)
5. PyTorch Dataset wrapper