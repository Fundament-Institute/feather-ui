
using Microsoft.Win32;
using System.Text.RegularExpressions;
using uniffi.calc;

public class Impl : uniffi.calc.Calculator
{
    private double last;
    private double? cur;
    private List<int> digits;
    private List<int> decimals;
    private bool decimal_mode;
    private CalcOp cur_op;

    public Impl()
    {
        last = 0.0;
        cur = null;
        digits = new List<int>();
        decimals = new List<int>();
        decimal_mode = false;
        cur_op = CalcOp.None;
    }
    void update_cur()
    {
        var x = 0.0;
        var mul = 1.0;
        digits.Reverse();
        foreach (var v in digits) {
            x += v * mul;
            mul *= 10;
        }
        digits.Reverse();

        mul = 0.1;
        foreach (var v in decimals)
        {
            x += v * mul;
            mul /= 10;
        }
        cur = x;
    }

    void apply_op()
    {
        if(cur != null) {
            var x = cur ?? 0;
            switch(cur_op) {
                case CalcOp.Add:
                    last = last + x;
                    break;
                case CalcOp.Sub:
                    last = last - x;
                    break;
                case CalcOp.Mul:
                    last = last * x;
                    break;
                case CalcOp.Div:
                    last = last / x;
                    break;
                case CalcOp.Mod:
                    last = last % x;
                    break;
                case CalcOp.Pow:
                    last = Math.Pow(last, x);
                    break;
                default:
                    break;
            }
        }

        switch(cur_op)
        {
            case CalcOp.Square:
                last = last * last;
                break;
            case CalcOp.Sqrt:
                last = Math.Sqrt(last);
                break;
            case CalcOp.Inv:
                last = Math.ReciprocalEstimate(last);
                break;
            case CalcOp.Negate:
                last = -last;
                break;
            case CalcOp.Clear:
                last = 0.0;
                break;
        }

        cur = null;
        cur_op = CalcOp.None;
        decimal_mode = false;
        decimals.Clear();
        digits.Clear();
    }

    void Calculator.AddDigit(byte digit)
    {
        if (@digit == 0 && digits.Count == 0 && !decimal_mode)
        {
            return;
        }
        if (decimal_mode)
        {
            decimals.Add(@digit);
        }
        else
        {
            digits.Add(@digit);
        }
        update_cur();
    }

    void Calculator.ApplyOp()
    {
        apply_op();
    }

    void Calculator.Backspace()
    {
        if (decimal_mode)
        {
            decimals.RemoveAt(decimals.Count - 1);
        }
        else
        {
            digits.RemoveAt(decimals.Count - 1);
        }
        update_cur();
    }

    Calculator Calculator.Copy()
    {
        var self = new Impl();

        self.last = last;
        self.cur = cur;
        self.digits = digits;
        self.decimals = decimals;
        self.decimal_mode = decimal_mode;
        self.cur_op = cur_op;

        return self;
    }

    bool Calculator.Eq(Calculator rhs)
    {
        var v = rhs as Impl;

        if(v == null)
        {
            return false;
        }

        return v.last == last &&
        v.cur == cur &&
        v.digits == digits &&
        v.decimals == decimals &&
        v.decimal_mode == decimal_mode &&
        v.cur_op == cur_op;
    }

    double Calculator.Get()
    {
            return cur ?? last;
    }

    void Calculator.SetOp(CalcOp op)
    {
        switch(op) {
            case CalcOp.Square | CalcOp.Sqrt | CalcOp.Inv | CalcOp.Negate | CalcOp.Clear:
                cur_op = op;
                apply_op();
                break;
            default:
                apply_op();
                cur_op = op;
                break;
        }
    }

    void Calculator.ToggleDecimal()
    {
            decimal_mode = !decimal_mode;
    }
}

namespace calculator_cs
{
    internal class Program
    {
        static void Main(string[] args)
        {
            uniffi.calc.CalcMethods.Register(new Impl());
        }
    }
}
