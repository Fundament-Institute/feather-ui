using uniffi.calc;
using System;
using System.Collections.Generic;

public class Calculator
{
    private double last;
    private double? cur;
    private List<int> digits;
    private List<int> decimals;
    private bool decimal_mode;
    private CalcOp op;

    Calculator()
    {
        last = 0.0;
        cur = null;
        digits = new List<int>();
        decimals = new List<int>();
        decimal_mode = false;
        op = CalcOp.None;
    }
    void update_cur()
    {
    }
    void AddDigit(byte @digit)
    {
        if (@digit == 0 && digits.Count == 0 && !decimal_mode) {
            return;
        }
        if(decimal_mode) {
            decimals.Add(@digit);
        }
        else
        {
            digits.Add(@digit);
        }
        update_cur();
    }
    void ApplyOp()
    {

    }
    void Backspace()
    {
        if(decimal_mode)
        {
            decimals.RemoveAt(decimals.Count - 1);
        } else
        {
            digits.RemoveAt(decimals.Count - 1);
        }
        update_cur();
    }
    Calculator Copy()
    {
        var self = new Calculator();

        self.last = last;
        self.cur = cur;
        self.digits = digits;
        self.decimals = decimals;
        self.decimal_mode = decimal_mode;
        self.op = op;

        return self;
    }
    bool Eq(Calculator @rhs)
    {
        return rhs == this;
    }
    double Get()
    {
        return cur??last;
    }
    void SetOp(CalcOp @op)
    {

    }
    void ToggleDecimal()
    {
        decimal_mode = !decimal_mode;
    }
}