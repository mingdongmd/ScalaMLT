# ScalaMLT

This is a NARX adptive inverse modeller and controller implemented by SISO NARX with couple inputs and one MLP network.

����NARXģ�͵�����Ӧ��ģ�Ϳ��ƣ���һ��MLPʵ��SISO NARX���ܴ�������롣


##SMLP definition(SMLP����)

```scala
case class SMLP(var XinNum:Int=2, var YfbNum:Int=0,var hiddenNum:Int=5, var coupleNum:Int=0, var coupleInNum:Int=0,var MUinNum:Int=0,var MYfbNum:Int=0)
XinNum Int=2, order of the input x(t). Default:2.
YfbNum Int=2, order of the feedback y(t). Default:2.
hiddenNum Int=5, hidden layer's number of the MLP. Default:5.
coupleNum Int=0, couple input number of the model. Default:0.
coupleInNum Int=0, order of the couple. Default:0.
MUinNum Int=0, control input u(t) order of the NARX model for this inverse NARX Controller. Default:0.
MYfbNum Int=0, the feedback y(t) order of the NARX model for this inverse NARX Controller. Default:0.
```


###Usage(�÷�)(in SMLPDemo.scala and SMLPInverseDemo.scala)��

```scala
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
```

* In this project, a sliding mode(VSC) added to ragulate the learning rate and improve the learning speed by tuning the active function's parameters 
in addition to the weights and bias. 
* �����̼����˻�ģ��ṹ���Ƶ���ѧϰ�ʡ�ͬʱ����������Ĳ����Լ�Ȩϵ����ƫ�ã������ѧϰ�ٶȡ�


###Build and run(����������)
```bash
>sbt
> package
[info] Updating {file:/D:/workplace/scala/ScalaMLT/}root...
[info] Resolving jline#jline;2.12.1 ...
[info] Done updating.
[info] Compiling 3 Scala sources to D:\workplace\scala\ScalaMLT\target\scala-2.11\classes...
[warn] there was one deprecation warning; re-run with -deprecation for details
[warn] one warning found
[warn] Multiple main classes detected.  Run 'show discoveredMainClasses' to see the list
[info] Packaging D:\workplace\scala\ScalaMLT\target\scala-2.11\scalamlt_2.11-0.0.1.jar ...
[info] Done packaging.
[success] Total time: 29 s, completed 2016-11-10 21:40:15
> run
[warn] Multiple main classes detected.  Run 'show discoveredMainClasses' to see the list

Multiple main classes detected, select one to run:

 [1] scala.ml.supervised.nnet.SMLPDemo
 [2] scala.ml.supervised.nnet.SMLPInverseDemo

Enter number: 1

[info] Running scala.ml.supervised.nnet.SMLPDemo
```

##Paper(����)��

* http://www-isl.stanford.edu/~widrow/papers/c1997nonlinearadaptive.pdf
* http://mocha-java.uccs.edu/dossier/RESEARCH/1998thesis-.pdf
* http://cdmd.cnki.com.cn/Article/CDMD-10005-2004082362.htm
* http://xueshu.baidu.com/s?wd=paperuri%3A%289e7f62257b124a98b73f6cfdf1a86bb2%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fwww.docin.com%2Fp-904291125.html&ie=utf-8&sc_us=18076168798136707397
* http://xueshu.baidu.com/s?wd=paperuri%3A%289e7f62257b124a98b73f6cfdf1a86bb2%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fd.wanfangdata.com.cn%2FThesis%2FY612820&ie=utf-8&sc_us=18076168798136707397


