"""Training: online teacher-rollout distillation (the CLIs are scripts/train.py and scripts/train_turbo.py).

    trainer         what the two trainers share: arguments and YAML recipes, optimizer groups, schedule, set-up
    curriculum      which budget and which layout each step trains on
    rollout         the streaming teacher rollout buffer (rollout_turbo: for the Idol-turbo teacher)
    losses          the distillation objective
    data            the Dome-Objaverse training images
"""
