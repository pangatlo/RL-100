import torch
import torch.nn as nn

class RL100(nn.Module):
    """
    Main RL-100 framework class.
    Orchestrates the three stages: Imitation Learning, Iterative Offline RL, and Online RL.
    """
    def __init__(self):
        raise NotImplementedError

class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy backbone.
    Models the action distribution as a conditional diffusion process.
    Supports both single-step and action-chunking control modes.
    """
    def __init__(self):
        raise NotImplementedError

class ConsistencyModel(nn.Module):
    """
    Consistency Model for fast inference.
    Distilled from the Diffusion Policy for one-step generation.
    """
    def __init__(self):
        raise NotImplementedError

class ImitationLearning:
    """
    Stage 1: Imitation Learning (IL).
    Initializes the policy using behavior cloning on human demonstrations.
    """
    def __init__(self):
        raise NotImplementedError

class OfflineRL:
    """
    Stage 2: Iterative Offline RL.
    Improves the policy using offline data with PPO-style updates and IQL critics.
    Includes OPE gating using a Transition Model.
    """
    def __init__(self):
        raise NotImplementedError

class OnlineRL:
    """
    Stage 3: Online RL.
    Fine-tunes the policy with on-policy data using PPO and GAE.
    """
    def __init__(self):
        raise NotImplementedError

class VisionEncoder(nn.Module):
    """
    Visual Encoder.
    Processes RGB images or Point Clouds.
    Shared across all stages and frozen during RL fine-tuning.
    """
    def __init__(self):
        raise NotImplementedError

class Critic(nn.Module):
    """
    Critic Network (Value Function / Q-Function).
    Used for advantage estimation in Offline (IQL) and Online (GAE) RL.
    """
    def __init__(self):
        raise NotImplementedError

class TransitionModel(nn.Module):
    """
    Transition Model.
    Used for Offline Policy Evaluation (OPE) gating in the offline RL stage.
    """
    def __init__(self):
        raise NotImplementedError
