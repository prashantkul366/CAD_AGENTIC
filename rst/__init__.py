"""Requirement-Satisfaction Trajectories (RST) for text-to-CAD.

A CadQuery program is treated as a sequence of kernel states. Requirements
compiled from the prompt are evaluated on every state, which gives a
satisfaction matrix M[t, i]. The matrix localises failures to individual
operations, drives local repair, and yields step-level rewards, without any
target geometry.
"""
