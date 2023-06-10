# -*- coding: utf-8 -*-
import model.model as md
from pytest import raises


class Testcnt_ratio(object):
    def test_none(self):
        pos, neg = md.KLoss().cnt_ratios(100, 202, None)
        assert(pos, 100)
        assert(neg, 202)

    def test_equal_easy(self):
        pos, neg = md.KLoss().cnt_ratios(100, 100, 1)
        assert(pos, 100)
        assert(neg, 100)

    def test_equal_pos_over(self):
        pos, neg = md.KLoss().cnt_ratios(101, 100, 1)
        assert(pos, 100)
        assert(neg, 100)

    def test_equal_neg_over(self):
        pos, neg = md.KLoss().cnt_ratios(100, 101, 1)
        assert(pos, 100)
        assert(neg, 100)

    def test_ratio_less_than_one(self):
        pos, neg = md.KLoss().cnt_ratios(100, 100, 0.1)
        assert(pos, 10)
        assert(neg, 100)

    def test_ratio_more_than_one(self):
        pos, neg = md.KLoss().cnt_ratios(100, 100, 10)
        assert(pos, 100)
        assert(neg, 10)
