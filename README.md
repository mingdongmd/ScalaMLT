This is a NARX adptive inverse modeller and controller implemented by SISO NARX with one MLP network.

case class SMLP(var XinNum:Int=2, var YfbNum:Int=0,var hiddenNum:Int=5, var coupleNum:Int=0, var coupleInNum:Int=0,var MUinNum:Int=0,var MYfbNum:Int=0)
XinNum Int=2, order of the input x(t). Default:2.
YfbNum Int=2, order of the feedback y(t). Default:2.
hiddenNum Int=5, hidden layer's number of the MLP. Default:5.
coupleNum Int=0, couple input number of the model. Default:0.
coupleInNum Int=0, order of the couple. Default:0.
MUinNum Int=0, control input u(t) order of the NARX model for this inverse NARX Controller. Default:0.
MYfbNum Int=0, the feedback y(t) order of the NARX model for this inverse NARX Controller. Default:0.

Usage(in SMLPDemo.scala and SMLPInverseDemo.scala)£º

import breeze.linalg._
import breeze.plot._
import breeze.stats.distributions._

object SMLPDemo extends App {
  val rd=Uniform(-1.0,1.0)
  var net=SMLP( XinNum=20, YfbNum=5, hiddenNum=100, coupleNum=1, coupleInNum=10)
  net.learningRate=0.5
  net.initW("uniform", -0.01, 0.01, 1.0)
    
  var y=0.0
  var plotU=ListBuffer[Double]()
  var plotX=ListBuffer[Double]()
  var plotY=ListBuffer[Double]()
  var plotD=ListBuffer[Double]()

  var desire=0.0
  var u=0.0
  var u1=Queue[Double]()
  for(x<-0 to 100) u1+=0.0
  for(x<-0.0 to 20.0 by 0.01){
    val couple=0.2*math.sin(1.5*x)
    u=0.25*math.sin(0.5*x)
    u1+=u
    if(u1.length>100) u1.dequeue()
    desire= desire/(1+desire*desire)+u*u*u+couple+0.02*rd.draw()
    val ux=u1.toArray
    net.setXin(ux(50))
    net.setCoupleIn(Array(couple))
    y=net.forward(desire)
    net.upDateW()
    if(x>0.01){
      plotU.add(u)
      plotX.add(x)
      plotY.add(y)
      plotD.add(desire)
    }
  }

  val f = Figure()
  val p = f.subplot(0)
  p += plot(plotX, plotY)
  p += plot(plotX, plotD,'.')
  p += plot(plotX, plotU,'+')
  p.xlabel = "x axis"
  p.ylabel = "y axis"
  f.saveas("lines.png")
}






