"""Task-specific final validation shared by external and Harbor evaluators.

The Harbor task receives a generated copy of the evaluator implementation and
public development data. Authoritative validation runs outside the agent
workspace with the task's evaluation inputs. Task modules define their own
workload, agreement rule, and supported measurement path; unsupported workloads
are reported explicitly instead of being measured by an alternate client.
"""
