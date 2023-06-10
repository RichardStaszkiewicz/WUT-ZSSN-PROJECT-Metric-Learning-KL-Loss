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

    def test_data_more_then_one_limit(self):
        pos, neg = md.KLoss().cnt_ratios(2, 6, 4)
        assert(pos, 0)
        assert(neg, 0)

    def test_data_more_then_one(self):
        pos, neg = md.KLoss().cnt_ratios(2, 6, 2.5)
        assert(pos, 2)
        assert(neg, 5)

    def test_data_less_then_one(self):
        pos, neg = md.KLoss().cnt_ratios(2, 6, 0.25)
        assert(pos, 1)
        assert(neg, 4)

    def test_data_exact_one(self):
        pos, neg = md.KLoss().cnt_ratios(2, 6, 1)
        assert(pos, 2)
        assert(neg, 2)
