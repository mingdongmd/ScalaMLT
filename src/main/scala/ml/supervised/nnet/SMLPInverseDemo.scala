package scala.ml.supervised.nnet
/**
 * Copyright 2016 Mingdong.
 * Licensed under the LGPL.
 * A NARX adptive plant modeller and  inverse controller demo.
 * email: mdwkmail@yeah.net
 * @author Mingdong
 * @version 0.0.1
 **/

import scala.collection._
import scala.collection.mutable._
import scala.annotation._
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.sys._
import scala.util.control._
import scala.util._
import scala.util.matching._
import scala.concurrent._

import breeze.linalg._
import breeze.plot._
import breeze.stats.distributions._

object SMLPInverseDemo{
  def main(args:Array[String])={
  val rd=Uniform(-1.0,1.0)
  
  var net=SMLP( XinNum=3, YfbNum=1, hiddenNum=5, coupleNum=0, coupleInNum=0)
  net.learningRate=0.1
  net.initW("uniform", -0.05, 0.05, 1.0)
  
  var netInv=SMLP( XinNum=5, YfbNum=2, hiddenNum=20, coupleNum=0, coupleInNum=0, MUinNum=3, MYfbNum=1)
  netInv.learningRate=0.15
  netInv.initW("uniform", -0.05, 0.05, 1.0)
  //netInv.tuningLearningRate=false
  netInv.learningRateMax=0.35
  netInv.learningRateMin= 0.01
  
  net.tuningGMax=5.0
  net.tuningGMin= 0.01
  net.tuningKMax=2.0
  net.tuningKMin=0.75
  netInv.tuningGMax=5.0
  netInv.tuningGMin= 0.01
  netInv.tuningKMax=2.0
  netInv.tuningKMin=0.75
  /*net.tuningXh=false
  net.tuningY=false
  netInv.tuningXh=false
  netInv.tuningY=false
  
  net.YtuningG=0.9
  net.YtuningK=0.9
  net.setXhtuningG(0.9)
  net.setXhtuningK(0.9)
  

  netInv.YtuningG=0.9
  netInv.YtuningK=0.9
  netInv.setXhtuningG(0.9)
  netInv.setXhtuningK(0.9) */
  
  var y=0.0
  var plotX=ListBuffer[Double]()
  var plotI=ListBuffer[Double]()
  var plotU=ListBuffer[Double]()
  var plotY=ListBuffer[Double]()
  var plotD=ListBuffer[Double]()

  var desire1=0.0
  var desire=0.0
  var u=0.0
  var u1=0.0
  for(x<-0.0 to 5.0 by 0.001){
    u=1.5*rd.draw()
    u1=u
    desire=desire1/(1+desire1*desire1)+u*u*u
    desire1=desire
    net.setXin(u)
    y=net.forward(desire)
    net.upDateW()
    /*if(x>0.0){
        plotX.add(x)
        plotI.add(u)
        plotY.add(y)
        plotD.add(desire)
      }*/
  }


  for(x<-0.0 to 20.0 by 0.001){
    var xin=0.0
    xin=0.1*math.sin(1.0*x)+0.5+0.001*rd.draw()
    netInv.setXin(xin)
    u=netInv.forwardInv()
    desire=desire1/(1+desire1*desire1)+u*u*u
    desire1=desire
    net.setXin(u)
    y=net.forward(desire)

    net.calcModelD
    net.upDateW()
    
    //netInv.YUinD=net.YUinDM
    //netInv.YYfbD=net.YYfbDM
    netInv.connectModel(net)
    netInv.upDateInverseW(xin-y)
    
    if(x>5.0){//println("netInv.learningRate="+netInv.learningRate)
        plotX.add(x)
        plotI.add(xin)
        plotU.add(u)
        plotY.add(y)
        plotD.add(desire)
      }
    }
  
    val f = Figure()
    val p = f.subplot(0)
    //f.visible=false
    p += plot(plotX, plotY)
    p += plot(plotX, plotI,'.')
    //p += plot(plotX, plotU,'-')
    //p += plot(plotX, plotD,'+')
    p.xlabel = "x axis"
    p.ylabel = "y axis"
    //f.saveas("outputLines.png")
  }
}