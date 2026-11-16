#!/usr/bin/env python3
"""Integrate actual board telemetry over the matching completed-counter interval."""
import argparse
import bisect
import json
import math
from pathlib import Path


def midpoint(row):
    return (row['before_ns'] + row['after_ns']) / 2e9


def board_power(row):
    found=[]
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key=='power_consumption_watts':
                    found.append(float(item))
                else: visit(item)
        elif isinstance(value,list):
            for item in value: visit(item)
    visit(row['electrical'])
    if len(found)!=1 or not math.isfinite(found[0]) or found[0]<=0:
        raise ValueError('Expected exactly one positive actual board power_consumption_watts field')
    return found[0]


def integrate(points,start,end):
    if end<=start or len(points)<2:
        raise ValueError('Insufficient integration interval')
    times=[p[0] for p in points]
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Non-increasing telemetry timestamps')
    if times[0]>start or times[-1]<end:
        raise ValueError('Power samples do not bracket the counter interval')
    def interpolate(t):
        i=bisect.bisect_right(times,t)-1
        if i==len(points)-1:return points[i][1]
        t0,p0=points[i];t1,p1=points[i+1]
        return p0+(p1-p0)*(t-t0)/(t1-t0)
    clipped=[(start,interpolate(start))]+[(t,p) for t,p in points if start<t<end]+[(end,interpolate(end))]
    return sum((tb-ta)*(pa+pb)*.5 for (ta,pa),(tb,pb) in zip(clipped,clipped[1:]))


# V2 uses paired_analysis.py; old idle-subtracted analysis intentionally removed.
