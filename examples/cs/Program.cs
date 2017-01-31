using System;
using fgDotNet;

namespace fgExample_cs
{
  static class Program
  {
    /// <summary>
    /// The main entry point for the application.
    /// </summary>
    [STAThread]
    static void Main()
    {
      var root = new Root("fgDirect2D_d.dll");

      //fgRegisterFunction("statelistener", [](fgElement* self, const FG_Msg*) { fgElement* progbar = fgRoot_GetID(fgSingleton(), "#progbar"); progbar->SetValueF(self->GetValueF(0) / self->GetValueF(1), 0); progbar->SetText(cStrF("%i", self->GetValue(0))); });
      //fgRegisterFunction("makepressed", [](fgElement* self, const FG_Msg*) { self->SetText("Pressed!"); });

      using (Layout layout = new Layout())
      {
        layout.LoadFileXML("../media/feathertest.xml");
        root.LayoutLoad(layout);

        root.GetID("#tabfocus").Action();

        while (root.ProcessMessages()) ;
      }
    }
  }
}
