You are a senior ML systems engineer.

Design a production-ready repository for training and serving Stable Diffusion models on a single multi-GPU machine (4×4090).

The system must support:
	•	LoRA fine-tuning
	•	Optional full UNet fine-tuning
	•	Distributed training (DDP or FSDP)
	•	Checkpointing and resumable training
	•	Versioned model artifacts
	•	GPU-backed inference service
	•	REST API for image generation
	•	Multi-GPU inference workers (1 worker per GPU)
	•	Queue-based request handling
	•	Dynamic LoRA loading at inference time

Constraints:
	•	Clean separation between training and inference
	•	Modular, extensible architecture
	•	No unnecessary abstraction layers
	•	Optimized for maintainability and performance
	•	Designed for long-running server operation
	•	Runs in Docker with CUDA support

Provide:
	1.	Repository structure (with clear directory layout)
	2.	Core components and their responsibilities
	3.	End-to-end data flow (training and inference)
	4.	Training architecture (model wrapping, dataset handling, distributed strategy)
	5.	Inference architecture (API layer, worker layer, GPU management)
	6.	Model artifact lifecycle (creation, storage, loading)
	7.	Deployment approach (Docker + runtime setup)
	8.	GPU allocation strategy
	9.	Logging, monitoring, and failure handling
	10.	Tradeoffs and design decisions with justification

Output in a structured technical format.
Avoid beginner explanations.
Focus on system design depth and engineering clarity.
