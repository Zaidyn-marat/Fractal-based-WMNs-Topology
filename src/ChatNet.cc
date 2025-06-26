#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"
//#include "custom-olsr-helper.h"
#include "ns3/olsr-helper.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
//#include "ns3/v4ping-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Project-2");


//std::string phyMode ("VhtMcs3");
//std::string phyMode ("VhtMcs9");


std::string datarate= "100000000";

uint32_t numNodes= 85;
bool verbose = false;

//int seed= 4;
int mode= 0;

//Simulation Timing
float routingTime       = 2.0;          // time added to start for olsr to converge, seconds
double flowtime         = 20.0;           // total time each source will transmit for.
double sinkExtraTime    = 2.0;         // extra timer the last packet has to reach the sink, seconds
float totalTime         = routingTime + flowtime + sinkExtraTime; // total simulation time, seconds

int main(int argc, char *argv[])
{
    //ns3::SeedManager::SetSeed(seed);
    // disable fragmentation for frames below 3000 bytes
    //Config::SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold", StringValue ("3000"));

    // turn on/off RTS/CTS for frames below 3000 bytes
    bool enableCtsRts       = false;
    UintegerValue ctsThr = (enableCtsRts ? UintegerValue (100) : UintegerValue (5000));
    Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", ctsThr);


    NodeContainer c;
    c.Create (numNodes);

    //wifi - standard/modes
    WifiHelper wifi;
    if(verbose) { wifi.EnableLogComponents (); } // Turn on all Wifi logging
    
    
    wifi.SetStandard(WIFI_STANDARD_80211ac);

    //wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
    //                              "DataMode",StringValue (phyMode),
    //                              "ControlMode",StringValue (phyMode));
    

    // Add an upper mac and disable rate control
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    
    //wifi - Phy
    YansWifiPhyHelper wifiPhy;
    wifiPhy.Set("RxGain", DoubleValue(-1.7));
    wifiPhy.Set("TxGain", DoubleValue(-1.7));  //250m LogDistancePropagationLossModel

    // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
    bool tracing = false;
    if(tracing) wifiPhy.SetPcapDataLinkType (YansWifiPhyHelper::DLT_IEEE802_11_RADIO);


    //wifi - Channel
    //YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    double logDropOff         = 2.0;
    bool useFriisDropoff     = false;
    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
    if(useFriisDropoff) {
        wifiChannel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
    }
    else {
        wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(logDropOff));
    }
    wifiPhy.SetChannel (wifiChannel.Create ());



    //wifi - Install
    NetDeviceContainer devices;
    devices = wifi.Install (wifiPhy, wifiMac, c);
    //bool tracing = false;
    if(tracing) {
        wifiPhy.EnablePcap("wifi-adhoc", devices);
        wifiPhy.EnableAscii("wifi-adhoc", devices);
    }

    //mobility - Grid Allocator
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

    // Open the file for reading
    std::ifstream inputFile("D10N100.txt");
    if (!inputFile.is_open()) {
        NS_FATAL_ERROR("Error opening the file!");
    }
    // Define vectors to store the data from each column
    std::vector<double> column1;
    std::vector<double> column2;
    // Read each line of the file
    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        double value1, value2;
        // Assuming the values are floating-point numbers separated by a space
        if (iss >> value1 >> value2) {
            column1.push_back(value1);
            column2.push_back(value2);
        } else {
            NS_LOG_ERROR("Error reading line: " << line);
        }
    }
    // Close the file
    inputFile.close();
    // Display the read data (for demonstration purposes)
    double x; 
    double y;
    for (size_t i = 0; i < column1.size(); ++i) {
        x=column1[i]*500;
        y=column2[i]*500;
        NS_LOG_UNCOND("x: " << x << "\ty: " << y );
        //NS_LOG_UNCOND("N: " << column1.size() );
        positionAlloc->Add (Vector (x, y, 0.0));//1
    }




    //uint32_t gridSize = 50; 
    //mobility.SetPositionAllocator("ns3::GridPositionAllocator",
    //                              "MinX", DoubleValue(0.0),
    //                              "MinY", DoubleValue(0.0),
    //                              "DeltaX", DoubleValue(50.0),
    //                              "DeltaY", DoubleValue(50.0),
    //                              "GridWidth", UintegerValue(gridSize),
    //                              "LayoutType", StringValue("RowFirst"));
    
    
    //positionAlloc->Add (Vector (0.0, 0.0, 0.0));
    //positionAlloc->Add (Vector (0.0, 80.0, 0.0));
    //positionAlloc->Add (Vector (0.0, 160.0, 0.0));
    //positionAlloc->Add (Vector (0.0, 240.0, 0.0));
    mobility.SetPositionAllocator (positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(c);


    // Enable OLSR
    OlsrHelper olsr;
    Ipv4StaticRoutingHelper staticRouting;
    Ipv4ListRoutingHelper list;
    list.Add(olsr, 100);
    list.Add(staticRouting, 0);
    


    // Ipv4
    InternetStackHelper internet;
    internet.SetRoutingHelper(list);
    internet.Install (c);

    
    Ipv4AddressHelper ipv4;
    NS_LOG_INFO ("Assign IP Addresses.");
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer i;
    i = ipv4.Assign (devices);


    for (uint32_t ii = 0; ii < c.GetN(); ++ii) {
    Ptr<Ipv4> ipv4 = c.Get(ii)->GetObject<Ipv4>();
    Ipv4InterfaceAddress iface = ipv4->GetAddress(1, 0);
    std::cout << "Node " << ii << " IP Address: " << iface.GetLocal() << std::endl;
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    //FlowMonitorHelper flowmon;
    FlowMonitorHelper* flowmon;
    Ptr<FlowMonitor> flowMonitor;
    flowmon = new FlowMonitorHelper();
    flowMonitor = flowmon->InstallAll();




    
     
    ApplicationContainer app;
    int port = 9;

    uint32_t s=0;
    std::set<std::pair<uint32_t, uint32_t>> usedPairs;
    Ptr<UniformRandomVariable> rand = CreateObject<UniformRandomVariable>(); 
    uint32_t Count = 0;
    while (Count < 100) {
        uint32_t src = rand->GetInteger(0, numNodes - 1);
        uint32_t dst = rand->GetInteger(0, numNodes - 1);
         if (src != dst && usedPairs.find({src, dst}) == usedPairs.end()) {
            usedPairs.insert({src, dst});
            NS_LOG_UNCOND("src: " << src << "\tdst: " << dst);
            //uint32_t starttime = rand->GetInteger(2, 10);
            //uint32_t stoptime = rand->GetInteger(10, 22); 
            //V4PingHelper ping(i.GetAddress(dst));
            //ping.SetAttribute("Verbose", BooleanValue(true));
            //ApplicationContainer app = ping.Install(c.Get(src));
            //app.Start(Seconds(1.0 + pingCount * 0.1));
            //app.Stop(Seconds(20.0));
            
            OnOffHelper onoff = OnOffHelper("ns3::UdpSocketFactory", Address(InetSocketAddress(i.GetAddress(dst), port)));
            onoff.SetConstantRate(DataRate(datarate));
            onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
            app = onoff.Install(c.Get(src));
            app.Start(Seconds(1+s*6));
            app.Stop(Seconds(6+s*6));
            //PacketSinkHelper sink = PacketSinkHelper("ns3::UdpSocketFactory", Address(InetSocketAddress(i.GetAddress(src), port)));
            //ApplicationContainer sinkApp = sink.Install(c.Get(dst));
            //sinkApp.Start(Seconds(1.0 + pingCount * 0.2));
            //sinkApp.Stop(Seconds(routingTime + flowtime + sinkExtraTime));

            s+=1;
            Count++;
        }
    }

    //int port = 9;
    //int sourceNode = 0;
    //int sinkNode = 3;
    //ApplicationContainer app;
    //OnOffHelper onoff = OnOffHelper("ns3::UdpSocketFactory", Address(InetSocketAddress(i.GetAddress(sinkNode), port)));
    //onoff.SetConstantRate(DataRate(datarate));
    //onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    //onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    //app = onoff.Install(c.Get(sourceNode));
    //app.Start(Seconds(routingTime));
    //app.Stop(Seconds(routingTime + flowtime));
    //PacketSinkHelper sink = PacketSinkHelper("ns3::UdpSocketFactory", Address(InetSocketAddress(i.GetAddress(sinkNode), port)));
    //ApplicationContainer sinkApp = sink.Install(c.Get(sinkNode));
    //sinkApp.Start(Seconds(routingTime+1));
    //sinkApp.Stop(Seconds(routingTime + flowtime + sinkExtraTime));

    


    //uint32_t s=0;
    //for (uint32_t sourceNode = 0; sourceNode < numNodes; sourceNode++)
    //  {
    //  for (uint32_t sinkNode = sourceNode; sinkNode < numNodes; sinkNode++)
    //    {
    //    if (sinkNode!=sourceNode)
    //      { 
    //      NS_LOG_UNCOND("sourceNode: " << sourceNode << "\ti: " << sinkNode);
    //      OnOffHelper onoff = OnOffHelper("ns3::UdpSocketFactory", Address(InetSocketAddress(i.GetAddress(sinkNode), sourceNode*3+sinkNode+10)));
    //      onoff.SetConstantRate(DataRate(datarate));
    //      app = onoff.Install(c.Get(sourceNode));
    //      app.Start(Seconds(routingTime));
    //      app.Stop(Seconds(routingTime + flowtime));
    

    //      PacketSinkHelper sink = PacketSinkHelper("ns3::UdpSocketFactory", Address(InetSocketAddress(i.GetAddress(sinkNode), sourceNode*3+sinkNode+10)));
    //      app.Start(Seconds(routingTime));
    //      app.Stop(Seconds(routingTime + flowtime + sinkExtraTime));
    //      }
    //    }
    //  }


   
    Simulator::Stop (Seconds(7+s*6));
    Simulator::Run ();
    Simulator::Destroy ();



    std::ofstream outputFile("ChatNet10.txt");

    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }


    //monitor dropped packets
    flowMonitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon->GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = flowMonitor->GetFlowStats();
   
        

    int TxPackets;
    int RxPackets;
  
   

    for(std::map<FlowId, FlowMonitor::FlowStats>::const_iterator iter = stats.begin(); iter != stats.end(); ++iter) {
        
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(iter->first);
        
        double throughput = iter->second.rxBytes * 8.0 / (iter->second.timeLastRxPacket.GetSeconds() - iter->second.timeFirstTxPacket.GetSeconds()) / 1024/1024;


        double delay = iter->second.delaySum.GetSeconds() / iter->second.rxPackets;
        NS_LOG_UNCOND("Delay: " << delay << " s");
        double jitter = iter->second.jitterSum.GetSeconds() / iter->second.rxPackets;
        NS_LOG_UNCOND("Jitter: " << jitter << " s");
        TxPackets=iter->second.txPackets;
        RxPackets=iter->second.rxPackets;
        NS_LOG_UNCOND("Flow ID: " << iter->first << " Src_Addr: " << t.sourceAddress << " Dst_Addr: " << t.destinationAddress);
        NS_LOG_UNCOND("Tx Packets = " << iter->second.txPackets);
        NS_LOG_UNCOND("Rx PAckets = " << iter->second.rxPackets);
        //NS_LOG_UNCOND("Dropped Packets = " << iter->second.txPackets - iter->second.rxPackets);
        NS_LOG_UNCOND("Throughput: " << throughput << " Mbps");
        
        outputFile <<throughput<<"\n"<< std::endl;
        outputFile <<TxPackets<<"\n"<< std::endl;
        outputFile <<RxPackets<<"\n"<< std::endl;
        outputFile <<delay<<"\n"<< std::endl;
        outputFile <<jitter<<"\n"<< std::endl;
        





        
    }
    
    outputFile.close();

    return 0;
}

    
    



