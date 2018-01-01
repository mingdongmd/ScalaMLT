package scala.ml.supervised.nnet

/**
 * Copyright 2016 Mingdong.
 * Licensed under the LGPL.
 * A NARX adptive inverse modeller and controller.
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

import java.io.File
import org.slf4j.LoggerFactory
import java.util.Date

import breeze.linalg._
import breeze.stats.distributions._

/**
 * A SISO NARX with one MLP network. 
 * @author Mingdong
 * 
 * @param XinNum Int=2, order of the input x(t). Default:2.
 * @param YfbNum Int=2, order of the feedback y(t). Default:2.
 * @param hiddenNum Int=5, hidden layer's number of the MLP. Default:5.
 * @param coupleNum Int=0, couple input number of the model. Default:0.
 * @param coupleInNum Int=0, order of the couple. Default:0.
 * @param MUinNum Int=0, control input u(t) order of the NARX model for this inverse NARX Controller. Default:0.
 * @param MYfbNum Int=0, the feedback y(t) order of the NARX model for this inverse NARX Controller. Default:0.
 */
case class SMLP(var XinNum:Int=2, var YfbNum:Int=0,var hiddenNum:Int=5, var coupleNum:Int=0, var coupleInNum:Int=0,var MUinNum:Int=0,var MYfbNum:Int=0) extends Serializable{
  //Model
  /**Xin(0) is bias.*/
  var Xin=DenseVector.zeros[Double](XinNum)
  Xin(0)=1.0
  var Win=DenseMatrix.zeros[Double](hiddenNum,XinNum)
  var Win1=DenseMatrix.zeros[Double](hiddenNum,XinNum)
  
  var couple=new Array[DenseVector[Double]](coupleNum)
  var coupleW=new Array[DenseMatrix[Double]](coupleNum)
  if(coupleInNum>0 && coupleNum>0) {
    for(i <- 0 until coupleNum) {
      couple(i)=DenseVector.zeros[Double](coupleInNum)
      coupleW(i)=DenseMatrix.zeros[Double](hiddenNum,coupleInNum)
    }
  }
  
  var Yfb=DenseVector.zeros[Double](YfbNum)
  var YfbWyfbD=DenseMatrix.zeros[Double](hiddenNum,YfbNum)
  var YfbWyfbD1=new Array[DenseMatrix[Double]](hiddenNum)
  if(YfbNum>0) {
    for(i <- 0 until hiddenNum) {
      YfbWyfbD1(i)=DenseMatrix.zeros[Double](YfbNum,YfbNum)
    }
  }
  var Wyfb=DenseMatrix.zeros[Double](hiddenNum,YfbNum)
    
  /**Xhidden(0) is bias.*/
  var Xhidden=DenseVector.zeros[Double](hiddenNum)
  var Shidden=DenseVector.zeros[Double](hiddenNum)
  var Vhidden=DenseVector.zeros[Double](hiddenNum)
  var XhtuningG=DenseVector.ones[Double](hiddenNum)
  var XhtuningK=DenseVector.ones[Double](hiddenNum)
  Xhidden(0)=1.0
  var derivativeXhidden=DenseVector.zeros[Double](hiddenNum)

  var Desire=0.0
  var Yout=0.0
  var Sout=0.0
  var Vout=0.0
  var YtuningG=1.0
  var YtuningK=1.0
  var derivativeYout=0.0
  var Wy=DenseVector.zeros[Double](hiddenNum)
  var Wy1=DenseVector.zeros[Double](hiddenNum)
  
  var error=0.0
  var tuningXh=true
  var tuningY=true
  var learningRate=0.01
  var learningRateMax=0.35
  var learningRateMin=0.01
  var tuningLearningRate=true
  var activation="tanh"
  var errorType="MSE"
  
  var tuningGMax=1.732
  var tuningGMin=1.0
  var tuningKMax=1.732
  var tuningKMin=1.0
  
  var inQueue:Queue[Double]=Queue[Double]()
  var outQueue:Queue[Double]=Queue[Double]()
  var coupleQueue=new Array[Queue[Double]](coupleNum)
  for(i <- 0 until coupleNum){
    coupleQueue(i)=mutable.Queue[Double]()
  }
  
  var YUinDM=DenseVector.zeros[Double](XinNum)
  var YYfbDM=DenseVector.zeros[Double](YfbNum)  
  
  //Inverse model
  var YUinD=DenseVector.zeros[Double](MUinNum)
  var UWyD=DenseVector.zeros[Double](hiddenNum)
  var UWyD1=DenseMatrix.zeros[Double](hiddenNum,MUinNum)
  var UWinD=DenseMatrix.zeros[Double](hiddenNum,XinNum)
  var UWinD1=new Array[DenseMatrix[Double]](MUinNum)
  if(XinNum>0 && MUinNum>0) {
    for(i <- 0 until MUinNum) {
      UWinD1(i)=DenseMatrix.zeros[Double](hiddenNum,XinNum)
    }
  }
  var UWyfbD=DenseMatrix.zeros[Double](hiddenNum,YfbNum)
  var UWyfbD1=new Array[DenseMatrix[Double]](MUinNum)
  if(YfbNum>0 && MUinNum>0) {
    for(i <- 0 until MUinNum) {
      UWyfbD1(i)=DenseMatrix.zeros[Double](hiddenNum,YfbNum)
    }
  }
  var UWcoupleD=new Array[DenseMatrix[Double]](coupleNum)
  if(coupleInNum>0 && coupleNum>0) {
    for(i <- 0 until coupleNum) {
      UWcoupleD(i)=DenseMatrix.zeros[Double](hiddenNum,coupleInNum)
    }
  }
  var UWcoupleD1=new Array[Array[DenseMatrix[Double]]](MUinNum)
  if(coupleInNum>0 && coupleNum>0 && MUinNum>0) {
    for(i <- 0 until MUinNum) {
      UWcoupleD1(i)=new Array[DenseMatrix[Double]](coupleNum)
      for(j<- 0 until coupleNum) UWcoupleD1(i)(j)=DenseMatrix.zeros[Double](hiddenNum,coupleInNum)
    }
  }
  
  var YYfbD=DenseVector.zeros[Double](MYfbNum)
  var YWyD=DenseVector.zeros[Double](hiddenNum)
  var YWyD1=DenseMatrix.zeros[Double](hiddenNum,MYfbNum)
  var YWinD=DenseMatrix.zeros[Double](hiddenNum,XinNum)
  var YWinD1=new Array[DenseMatrix[Double]](MYfbNum)
  if(XinNum>0 && MYfbNum>0) {
    for(i <- 0 until MYfbNum) {
      YWinD1(i)=DenseMatrix.zeros[Double](hiddenNum,XinNum)
    }
  }
  var YWyfbD=DenseMatrix.zeros[Double](hiddenNum,YfbNum)
  var YWyfbD1=new Array[DenseMatrix[Double]](MYfbNum)
  if(YfbNum>0 && MYfbNum>0) {
    for(i <- 0 until MYfbNum) {
      YWyfbD1(i)=DenseMatrix.zeros[Double](hiddenNum,YfbNum)
    }
  }
  var YWcoupleD=new Array[DenseMatrix[Double]](coupleNum)
  if(coupleInNum>0 && coupleNum>0) {
    for(i <- 0 until coupleNum) {
      YWcoupleD(i)=DenseMatrix.zeros[Double](hiddenNum,coupleInNum)
    }
  }
  var YWcoupleD1=new Array[Array[DenseMatrix[Double]]](MYfbNum)
  if(coupleInNum>0 && coupleNum>0 && MYfbNum>0) {
    for(i <- 0 until MYfbNum) {
      YWcoupleD1(i)=new Array[DenseMatrix[Double]](coupleNum)
      for(j<- 0 until coupleNum) UWcoupleD1(i)(j)=DenseMatrix.zeros[Double](hiddenNum,coupleInNum)
    }
  }

  var UYtuningGD1=DenseVector.zeros[Double](MUinNum)
  var UYtuningKD1=DenseVector.zeros[Double](MUinNum)
  var UXhtuningGD=DenseVector.ones[Double](hiddenNum)
  var UXhtuningKD=DenseVector.ones[Double](hiddenNum)
  var UXhtuningGD1=DenseMatrix.ones[Double](hiddenNum,MUinNum)
  var UXhtuningKD1=DenseMatrix.ones[Double](hiddenNum,MUinNum)
  var YYtuningGD=0.0
  var YYtuningKD=0.0
  var YYtuningGD1=DenseVector.zeros[Double](MYfbNum)
  var YYtuningKD1=DenseVector.zeros[Double](MYfbNum)
  var YXhtuningGD=DenseVector.ones[Double](hiddenNum)
  var YXhtuningKD=DenseVector.ones[Double](hiddenNum)
  var YXhtuningGD1=DenseMatrix.ones[Double](hiddenNum,MYfbNum)
  var YXhtuningKD1=DenseMatrix.ones[Double](hiddenNum,MYfbNum)
  
  //Methods
  //Model methods
/**
 *  Set input x(t) to this NARX plant model.
 * @param x Double=0.0, the input x(t). Default:0.0.
 */
  def setXin(x:Double=0.0)={
    if(XinNum>1){
      if(inQueue.length>=XinNum-1) {
        inQueue.dequeue()
        inQueue+=x
      }else{
        inQueue.dequeueAll { t => true }
        inQueue+=x
        if(XinNum>2) for(i<- 0 to XinNum-3){
          inQueue+=0.0
        }
      }
      Xin(1 until XinNum):=DenseVector[Double](inQueue.toArray)
    }
  }
  
/**
 *  Set input x(t) to this NARX plant model.
 * @param x Double=0.0, the input x(t). Default:0.0.
 */
  def putXin(x:Double=0.0)={
    val xVect=Xin.toArray.drop(1)
    Xin=DenseVector[Double](xVect++Array(x))
  }
  
/**
 *  Set model output y(t) to the Queue for feedback of this NARX plant model.
 * @param y Double=0.0, model output y(t). Default:0.0.
 */
  def setYout(y:Double=0.0)={
    if(YfbNum>0){
      if(outQueue.length>=YfbNum) {
        outQueue.dequeue()
        outQueue+=y
      }else{
        outQueue.dequeueAll { t => true }
        outQueue+=y
        if(YfbNum>1) for(i<- 0 to YfbNum-2){
          outQueue+=0.0
        }
      }
      Yfb:=DenseVector[Double](outQueue.toArray)
    }
  }

/**
 *  Set model output y(t) to the Queue for feedback of this NARX plant model.
 * @param y Double=0.0, model output y(t). Default:0.0.
 */
  def putYout(y:Double=0.0)={
    val yVect=Yfb.toArray.drop(1)
    Yfb=DenseVector[Double](yVect++Array(y))
  }
  
/**
 *  Set couple input C(t) to this NARX plant model.
 * @param coupleIn Array[Double], the couple input C(t).
 */  
  def setCoupleIn(coupleIn:Array[Double])={
    if(coupleNum>0){
      for(i <- 0 until coupleNum){
        if(coupleQueue(i).length>=coupleInNum) {
          coupleQueue(i).dequeue()
          coupleQueue(i)+=coupleIn(i)
        }else{
          coupleQueue(i).dequeueAll { t => true }
          coupleQueue(i)+=coupleIn(i)
          if(coupleInNum>1) for(j<- 0 to coupleInNum-2){
            coupleQueue(i)+=0.0
          }
        }
        couple(i):=DenseVector[Double](coupleQueue(i).toArray)
      }
    }
  }
  
/**
 *  Set couple input C(t) to this NARX plant model.
 * @param coupleIn Array[Double], the couple input C(t).
 */  
  def putCoupleIn(coupleIn:Array[Double])={
    if(coupleNum>0){
      for(i <- 0 until coupleNum){
        val coupleInput=coupleIn(i)
        val ins=couple(i).toArray.drop(1)
        couple(i):=DenseVector[Double](ins++Array(coupleInput))
      }
    }
  }
  
  
/**
 *  Set model's hidden layer tuning parameter K, that is sigmoid(K(Wx+b)).
 * @param k Double=1.0, tuning parameter K. Default:1.0.
 */
  def setXhtuningK(k:Double=1.0)={
      for(i <- 1 until hiddenNum) {
      XhtuningK(i)=k
    }
  }
  
/**
 *  Set model's hidden layer tuning parameter G, that is sigmoid(K(Wx+b)).
 * @param g Double=1.0, tuning parameter G. Default:1.0.
 */
  def setXhtuningG(g:Double=1.0)={
      for(i <- 1 until hiddenNum) {
      XhtuningG(i)=g
    }
  }
    
  def logistic(v:Double=0.0,a:Double=1.0)={
    1.0/(1.0+math.exp(-a*v))
  }

/**
 * Get activate of sum s with tuning parameter K.
 * @param tuningK Double=1.0, tuning parameter K. Default:1.0.
 * @param s Double=1.0, sum s(t).
 * @return the activate.
 */
  def activate(tuningK:Double=1.0,s:Double)={
    activation match{
      case "tanh" =>  math.tanh(tuningK*s)    //derivativeTanh=1-v*v
    }
  }

/**
 * Forward computing.
 * @param desire Double=0.0, the desire output. Default:1.0.
 * @return the model's output.
 */
  def forward(desire:Double=0.0)={//YfbNum:Int=1,var hiddenNum:Int=5, var coupleNum:Int=0, var coupleInNum:Int=0
    Desire=desire
    var sum=Win*Xin  //One row weights belong to a  hidden Neuron
    if(YfbNum!=0) sum=sum+Wyfb*Yfb
    if(coupleNum!=0) for(i <- 0 until coupleNum) sum=sum+coupleW(i)*couple(i)
    
    for(i <- 1 until hiddenNum) {
      Shidden(i)=sum(i)
      Vhidden(i)=activate(XhtuningK(i),Shidden(i))
      Xhidden(i)=XhtuningG(i)*Vhidden(i)
      derivativeXhidden(i)=1-Vhidden(i)*Vhidden(i)
    }
    activation match{
      case "tanh" => {
        Sout=Wy.t*Xhidden
        Vout=activate(YtuningK,Sout)
        Yout=YtuningG*Vout
        derivativeYout=1-Vout*Vout
      }
    }
    error=Desire-Yout
    //setYout(Yout)
    putYout(Yout)
    Yout
  }
  
  def getCost()={
    errorType match{
      case "MSE" => (Desire-Yout)*(Desire-Yout)/2.0
      case _ => Double.MaxValue
    }
  }
  
/**
 * Update model's weights.
 */
  def upDateW()={
    errorType match{
      case "MSE" => {
        activation match{
          case "tanh" =>{
            val delt=2*learningRate*error*YtuningG*derivativeYout*YtuningK
            Wy1=Wy
            Wy=Wy1+delt*Xhidden
            for(i <- 1 until hiddenNum) {
              val upWin=delt*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*XhtuningK(i)*Xin.t+Win(i, ::)
              Win(i,::) := upWin(0,::)
              if(YfbNum>1){
                for(j<- YfbNum-1 until 0 by -1){
                  YfbWyfbD1(i)(j,::) := YfbWyfbD1(i)(j-1,::)
                }
                YfbWyfbD(i, ::) := (Yfb.t+Wyfb(i, ::)*YfbWyfbD1(i))*YtuningG*derivativeYout*YtuningK*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*XhtuningK(i)
                YfbWyfbD1(i)(0, ::) := YfbWyfbD(i, ::)
                val upWyfb=2*learningRate*error*YfbWyfbD(i,::)
                Wyfb(i,::) := Wyfb(i,::)+upWyfb(0,::)
              }
              if(coupleNum>0){
                for(j <- 0 until coupleNum) {
                    val upCouple=delt*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*XhtuningK(i)*couple(j).t+coupleW(j)(i,::)
                    coupleW(j)(i,::) :=upCouple(0,::)
                }
              }
            }
            if(tuningY){
              YtuningG=YtuningG+2*learningRate*error*Vout
              YtuningK=YtuningK+2*learningRate*error*YtuningG*derivativeYout*(Wy1.t*Xhidden)
              if(math.abs(YtuningG)>tuningGMax) YtuningG=tuningGMax*math.signum(YtuningG)
              if(math.abs(YtuningG)<tuningGMin) YtuningG=tuningGMin*math.signum(YtuningG)
              if(math.abs(YtuningK)>tuningKMax) YtuningK=tuningKMax*math.signum(YtuningK)
              if(math.abs(YtuningK)<tuningKMin) YtuningK=tuningKMin*math.signum(YtuningK)
            }
            if(tuningXh){
              for(i <- 1 until hiddenNum) {
                XhtuningG(i)=XhtuningG(i)+2*learningRate*error*YtuningG*derivativeYout*YtuningK*Wy1(i)*Vhidden(i)
                XhtuningK(i)=XhtuningK(i)+2*learningRate*error*YtuningG*derivativeYout*YtuningK*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*Shidden(i)
                if(math.abs(XhtuningG(i))>tuningGMax) XhtuningG(i)=tuningGMax*math.signum(XhtuningG(i))
                if(math.abs(XhtuningG(i))<tuningGMin) XhtuningG(i)=tuningGMin*math.signum(XhtuningG(i))
                if(math.abs(XhtuningK(i))>tuningKMax) XhtuningK(i)=tuningKMax*math.signum(XhtuningK(i))
                if(math.abs(XhtuningK(i))<tuningKMin) XhtuningK(i)=tuningKMin*math.signum(XhtuningK(i))
              }
            }
          }
          //case _ =>
        }
        
      }
      //case _ =>
    }
  }
  
  //Inverse model methed
/**
 * Computing model's derivatives for inverse control model.
 */
  def calcModelD={
    var XhUD=Win
    var XhYD=Wyfb
    for(i <- 0 until hiddenNum){
      val k=XhtuningG(i)*XhtuningK(i)*derivativeXhidden(i)
      XhUD(i,::) := XhUD(i,::)*k
      XhYD(i,::) :=XhYD(i,::)*k
    }
    val g=YtuningG*derivativeYout*YtuningK
    YUinDM=g*(Wy.t*XhUD).t
    YYfbDM=g*(Wy.t*XhYD).t
  }
  
  def setYUinD(modelYUinD:DenseVector[Double])={
    if(modelYUinD.length==YUinD.length) YUinD=modelYUinD  else println("modelYUinD.length!=YUinD.length")
  }
  
  def setYYfbD(modelYYfbD:DenseVector[Double])={
    if(modelYYfbD.length==YYfbD.length) YYfbD=modelYYfbD  else println("modelYYfbD.length!=YYfbD.length")
  }
  
/**
 * Connecting this inverse control model to plant model.
 * @param model SMLP, plant model.
 */
  def connectModel(model:SMLP)={
    if(model.YUinDM.length==YUinD.length) YUinD=model.YUinDM  else println("model.YUinDM.length!=YUinD.length")
    if(model.YYfbDM.length==YYfbD.length) YYfbD=model.YYfbDM  else println("model.YYfbDM.length!=YYfbD.length")
  }
  
  def forwardInv()={//YfbNum:Int=1,var hiddenNum:Int=5, var coupleNum:Int=0, var coupleInNum:Int=0
    var sum=Win*Xin  //One row weights belong to a  hidden Neuron
    if(YfbNum!=0) sum=sum+Wyfb*Yfb
    if(coupleNum!=0) for(i <- 0 until coupleNum) sum=sum+coupleW(i)*couple(i)
    
    for(i <- 1 until hiddenNum) {
      Shidden(i)=sum(i)
      Vhidden(i)=activate(XhtuningK(i),Shidden(i))
      Xhidden(i)=XhtuningG(i)*Vhidden(i)
      derivativeXhidden(i)=1-Vhidden(i)*Vhidden(i)
    }
    activation match{
      case "tanh" => {
        Sout=Wy.t*Xhidden
        Vout=activate(YtuningK,Sout)
        Yout=YtuningG*Vout
        derivativeYout=1-Vout*Vout
      }
    }
    //setYout(Yout)
    putYout(Yout)
    Yout
  }
  
/**
 * Update inverse controller's weights.
 * @param err Double, error between the desire and plant model's output.
 */
  def upDateInverseW(err:Double)={
    error=err
    errorType match{
      case "MSE" => {
        activation match{
          case "tanh" =>{
            val delt=2*learningRate*error
            val gDYk=YtuningG*derivativeYout*YtuningK
            var lrWD=0.0
            UWyD = gDYk*Xhidden
            for(i <- MUinNum-1 to 1 by -1) UWyD1(::,i) := UWyD1(::,i-1)
            UWyD1(::,0) := UWyD
            YWyD=UWyD1*YUinD+YWyD1*YYfbD
            Wy1=Wy
            Wy=Wy1+delt*YWyD
            if(tuningLearningRate) lrWD+=YWyD.map { x => x*x }.sum
              
            for(i <- 1 until hiddenNum) {
              val dUWin= gDYk*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*XhtuningK(i)*Xin.t
              UWinD(i,::) := dUWin(0,::)
              
              if(YfbNum>1){
                for(j<- YfbNum-1 until 0 by -1){
                  YfbWyfbD1(i)(j,::) := YfbWyfbD1(i)(j-1,::)
                }
                YfbWyfbD(i, ::) := (Yfb.t+Wyfb(i, ::)*YfbWyfbD1(i))*gDYk*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*XhtuningK(i)
                YfbWyfbD1(i)(0, ::) := YfbWyfbD(i, ::)
                UWyfbD(i,::) := YfbWyfbD(i,::)
              }
              
              if(coupleNum>0){
                for(j <- 0 until coupleNum) {
                    val dYWcouple=gDYk*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*XhtuningK(i)*couple(j).t
                    UWcoupleD(j)(i,::) :=dYWcouple(0,::)
                }
              }
            }
            
            for(i <- MUinNum-1 to 1 by -1) UWinD1(i)=UWinD1(i-1)
            UWinD1(0)=UWinD
            YWinD=YWinD*0.0
            for(i <- 0 until MUinNum) YWinD=YWinD+UWinD1(i)*YUinD(i)
            for(i <- 0 until MYfbNum) YWinD=YWinD+YWinD1(i)*YYfbD(i)
            
            for(i <- MUinNum-1 to 1 by -1) UWyfbD1(i)=UWyfbD1(i-1)
            UWyfbD1(0)=UWyfbD
            YWyfbD=YWyfbD*0.0
            for(i <- 0 until MUinNum) YWyfbD=YWyfbD+UWyfbD1(i)*YUinD(i)
            for(i <- 0 until MYfbNum) YWyfbD=YWyfbD+YWyfbD1(i)*YYfbD(i)
            
            for(i <- MUinNum-1 to 1 by -1) UWcoupleD1(i)=UWcoupleD1(i-1)
            UWcoupleD1(0)=UWcoupleD
            YWcoupleD=YWcoupleD.map { x => x*0.0 }
            for(i <- coupleNum-1 to 1 by -1) {
              for(j <- MUinNum-1 to 1 by -1) {
                YWcoupleD(i)=YWcoupleD(i)+UWcoupleD1(j)(i)*YUinD(i)
              }
            }
            for(i <- coupleNum-1 to 1 by -1) {
              for(j <- MYfbNum-1 to 1 by -1) {
                YWcoupleD(i)=YWcoupleD(i)+YWcoupleD1(j)(i)*YYfbD(i)
              }
            }
            
            for(i <- MYfbNum-1 to 1 by -1) {
              YWyD1(::,i) := YWyD1(::,i-1)
              YWinD1(i) := YWinD1(i-1)
              YWyfbD1(i)=YWyfbD1(i-1)
              YWcoupleD1(i)=YWcoupleD1(i-1)
            }
            YWyD1(::,0) := YWyD
            YWinD1(0)=YWinD
            YWyfbD1(0)=YWyfbD
            YWcoupleD1(0)=YWcoupleD
            ///////////////////////////////////////////////
            if(tuningY){
              for(i <- MUinNum-1 to 1 by -1) {
                UYtuningGD1(i)=UYtuningGD1(i-1)
                UYtuningKD1(i)=UYtuningKD1(i-1)
              }
              UYtuningGD1(0)=Vout
              UYtuningKD1(0)=YtuningG*derivativeYout*(Wy1.t*Xhidden)
              YYtuningGD=UYtuningGD1.t*YUinD+YYtuningGD1.t*YYfbD
              YYtuningKD=UYtuningKD1.t*YUinD+YYtuningKD1.t*YYfbD
              
              for(i <- MYfbNum-1 to 1 by -1) {
                YYtuningGD1(i)=YYtuningGD1(i-1)
                YYtuningKD1(i)=YYtuningKD1(i-1)
              }
              YYtuningGD1(0)=YYtuningGD
              YYtuningKD1(0)=YYtuningKD
            }
            if(tuningXh){
              for(i <- MUinNum-1 to 1 by -1) {
                UXhtuningGD1(::,i):=UXhtuningGD1(::,i-1)
                UXhtuningKD1(::,i):=UXhtuningKD1(::,i-1)
              }
              for(i <- 1 until hiddenNum) {
                UXhtuningGD1(i,0)=gDYk*Wy1(i)*Vhidden(i)
                UXhtuningKD1(i,0)=gDYk*Wy1(i)*XhtuningG(i)*derivativeXhidden(i)*Shidden(i)
              }
              YXhtuningGD=UXhtuningGD1*YUinD+(YXhtuningGD1*YYfbD)
              YXhtuningKD=UXhtuningKD1*YUinD+(YXhtuningKD1*YYfbD)
              for(i <- MYfbNum-1 to 1 by -1) {
                YXhtuningGD1(::,i):=YXhtuningGD1(::,i-1)
                YXhtuningKD1(::,i):=YXhtuningKD1(::,i-1)
              }
              YXhtuningGD1(::,0):=YXhtuningGD
              YXhtuningKD1(::,0):=YXhtuningKD
            }
            ///////////////////////////////////////////////updata weight////////////////////////////////////
            //val delt=2*learningRate*error
            Wyfb=Wyfb+delt*YWyfbD
            Win=Win+delt*YWinD
            if(coupleInNum>0 && coupleNum>0) {
              for(i <- 0 until coupleNum) {
                coupleW(i)=coupleW(i)+delt*YWcoupleD(i)
              }
            }
            
            if(tuningLearningRate){
              for(i<- 0 until hiddenNum){
                lrWD+=YWyfbD(i, ::).inner.toArray.map { x => x*x }.sum
                lrWD+=YWinD(i, ::).inner.toArray.map { x => x*x }.sum
              }
              if(coupleInNum>0 && coupleNum>0) for(i<- 0 until coupleNum) lrWD += YWcoupleD(i).toArray.map { x => x*x }.sum
            }
            if(tuningY){
              YtuningG=YtuningG+delt*YYtuningGD
              YtuningK=YtuningK+delt*YYtuningKD
              if(math.abs(YtuningG)>tuningGMax) YtuningG=tuningGMax*math.signum(YtuningG)
              if(math.abs(YtuningG)<tuningGMin) YtuningG=tuningGMin*math.signum(YtuningG)
              if(math.abs(YtuningK)>tuningKMax) YtuningK=tuningKMax*math.signum(YtuningK)
              if(math.abs(YtuningK)<tuningKMin) YtuningK=tuningKMin*math.signum(YtuningK)
              if(tuningLearningRate) lrWD += YYtuningGD*YYtuningGD+YYtuningKD*YYtuningKD
            }
            if(tuningXh){
                XhtuningG=XhtuningG+delt*YXhtuningGD
                XhtuningK=XhtuningK+delt*YXhtuningKD
                for(i <- 1 until hiddenNum) {
                  if(math.abs(XhtuningG(i))>tuningGMax) XhtuningG(i)=tuningGMax*math.signum(XhtuningG(i))
                  if(math.abs(XhtuningG(i))<tuningGMin) XhtuningG(i)=tuningGMin*math.signum(XhtuningG(i))
                  if(math.abs(XhtuningK(i))>tuningKMax) XhtuningK(i)=tuningKMax*math.signum(XhtuningK(i))
                  if(math.abs(XhtuningK(i))<tuningKMin) XhtuningK(i)=tuningKMin*math.signum(XhtuningK(i))
                }
                if(tuningLearningRate) lrWD += YXhtuningGD.toArray.map { x => x*x }.sum+YXhtuningKD.toArray.map { x => x*x }.sum
            }
            if(tuningLearningRate) {
              learningRate=(Xin(0)-Xin(1)+0.02*math.signum(error)-0.03*error)/(4*lrWD*error)
              if(learningRate>learningRateMax)  learningRate=learningRateMax
              if(learningRate<learningRateMin)  learningRate=learningRateMin
            }
          }
          //case _ =>
        }
        
      }
      //case _ =>
    }
  }
  
/**
 * Initializing weights.
 * @param initType String, type of Initialization. normal, uniform, Gamma, Exponential, Laplace, Poisson. Default: normal.
 * @param m Double, Mean, low etc.. Default: 0.0.
 * @param s Double, Standard deviation, high etc.. Default: 0.01.
 * @param g Double, Multiplier gain. Default: 1.0.
 */
  def initW(initType:String="normal",m:Double=0.0,s:Double=0.01,g:Double=1.0)={
      initType match{
        case "normal" =>{
          val rd=Gaussian(m,s)
          for(i <-0 until Wy.length) Wy(i)=g*rd.get()
          for(i <-0 until Wy1.length) Wy1(i)=g*rd.get()
          for(i <- 0 until Win.rows){
            for(j <- 0 until Win.cols){
              Win(i,j)=g*rd.get()
              Win1(i,j)=g*rd.get()
            }
          }
          for(i <- 0 until Wyfb.rows){
            for(j <- 0 until Wyfb.cols){
              Wyfb(i,j)=g*rd.get()
            }
          }
          for(i <- 0 until coupleW.length){
            for(j <- 0 until coupleW(i).rows){
              for(k <- 0 until coupleW(i).cols){
                coupleW(i)(j,k)=g*rd.get()
              }
            }
          }
        }
        case "uniform" =>{
          val rd=Uniform(m,s)
          for(i <-0 until Wy.length) Wy(i)=g*rd.draw()
          for(i <-0 until Wy1.length) Wy1(i)=g*rd.draw()
          for(i <- 0 until Win.rows){
            for(j <- 0 until Win.cols){
              Win(i,j)=g*rd.draw()
              Win1(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until Wyfb.rows){
            for(j <- 0 until Wyfb.cols){
              Wyfb(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until coupleW.length){
            for(j <- 0 until coupleW(i).rows){
              for(k <- 0 until coupleW(i).cols){
                coupleW(i)(j,k)=g*rd.draw()
              }
            }
          }
        }
        case "Gamma" =>{
          val rd=Gamma(m,s)
          for(i <-0 until Wy.length) Wy(i)=g*rd.draw()
          for(i <-0 until Wy1.length) Wy1(i)=g*rd.draw()
          for(i <- 0 until Win.rows){
            for(j <- 0 until Win.cols){
              Win(i,j)=g*rd.draw()
              Win1(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until Wyfb.rows){
            for(j <- 0 until Wyfb.cols){
              Wyfb(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until coupleW.length){
            for(j <- 0 until coupleW(i).rows){
              for(k <- 0 until coupleW(i).cols){
                coupleW(i)(j,k)=g*rd.draw()
              }
            }
          }
        }
        case "Exponential" =>{
          val rd=Exponential(m)
          for(i <-0 until Wy.length) Wy(i)=g*rd.draw()
          for(i <-0 until Wy1.length) Wy1(i)=g*rd.draw()
          for(i <- 0 until Win.rows){
            for(j <- 0 until Win.cols){
              Win(i,j)=g*rd.draw()
              Win1(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until Wyfb.rows){
            for(j <- 0 until Wyfb.cols){
              Wyfb(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until coupleW.length){
            for(j <- 0 until coupleW(i).rows){
              for(k <- 0 until coupleW(i).cols){
                coupleW(i)(j,k)=g*rd.draw()
              }
            }
          }
        }
        case "Laplace" =>{
          val rd=Laplace(m,s)
          for(i <-0 until Wy.length) Wy(i)=g*rd.draw()
          for(i <-0 until Wy1.length) Wy1(i)=g*rd.draw()
          for(i <- 0 until Win.rows){
            for(j <- 0 until Win.cols){
              Win(i,j)=g*rd.draw()
              Win1(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until Wyfb.rows){
            for(j <- 0 until Wyfb.cols){
              Wyfb(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until coupleW.length){
            for(j <- 0 until coupleW(i).rows){
              for(k <- 0 until coupleW(i).cols){
                coupleW(i)(j,k)=g*rd.draw()
              }
            }
          }
        }
        case "Poisson" =>{
          val rd=Poisson(m)
          for(i <-0 until Wy.length) Wy(i)=g*rd.draw()
          for(i <-0 until Wy1.length) Wy1(i)=g*rd.draw()
          for(i <- 0 until Win.rows){
            for(j <- 0 until Win.cols){
              Win(i,j)=g*rd.draw()
              Win1(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until Wyfb.rows){
            for(j <- 0 until Wyfb.cols){
              Wyfb(i,j)=g*rd.draw()
            }
          }
          for(i <- 0 until coupleW.length){
            for(j <- 0 until coupleW(i).rows){
              for(k <- 0 until coupleW(i).cols){
                coupleW(i)(j,k)=g*rd.draw()
              }
            }
          }
        }
        //case _ =>
      }
  }
  
  def fit(xyData:Array[(Double,Double)])={
    xyData.foreach(data=> data match{
      case (x,y) =>{
        setXin(x)
        forward(y)
        upDateW
      }
    })
  }
  
}