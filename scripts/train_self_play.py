"""
train_self_play.py – Train AIRS defender and attacker via self-play.

Usage
-----
python scripts/train_self_play.py [--rounds 10] [--defender_steps 20000]
                                   [--attacker_steps 20000] [--seed 42]
                                   [--output_dir models/self_play]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.adversarial_attacker import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description="AIRS Self-Play Training")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--defender_steps", type=int, default=20_000)
    parser.add_argument("--attacker_steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="models/self_play")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    trainer = SelfPlayTrainer(
        rounds=args.rounds,
        defender_steps=args.defender_steps,
        attacker_steps=args.attacker_steps,
        seed=args.seed,
    )

    print(f"Starting self-play: {args.rounds} rounds, "
          f"defender={args.defender_steps:,} steps, "
          f"attacker={args.attacker_steps:,} steps")

    result = trainer.train()
    print(f"Self-play finished — {result['rounds']} rounds completed")

    defender_path = os.path.join(args.output_dir, "defender")
    attacker_path = os.path.join(args.output_dir, "attacker")
    trainer.save(defender_path, attacker_path)
    print(f"Models saved: defender → {defender_path}, attacker → {attacker_path}")


if __name__ == "__main__":
    main()
