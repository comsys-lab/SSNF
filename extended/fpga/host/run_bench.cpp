#include <xrt/xrt_device.h>
#include <experimental/xrt_ip.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <array>
namespace {
using Clock=std::chrono::steady_clock;
uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
std::vector<uint32_t> hex_words(const std::string& text,unsigned words){
 if(text.size()!=words*8)throw std::runtime_error("Malformed reference width");
 std::vector<uint32_t> out;for(unsigned i=0;i<words;i++)out.push_back(std::stoul(text.substr(text.size()-8*(i+1),8),nullptr,16));return out;
}
struct Row{unsigned mode,on,n,rep;std::vector<uint32_t> seed,state,record;uint32_t result;};
struct Controller{
 xrt::device device;xrt::uuid uuid;xrt::ip ip;std::string vectors;
 Controller(const std::string& bdf,const std::string& image,const std::string& dir):device(bdf),uuid(device.load_xclbin(image)),ip(device,uuid,"ssnf_bench"),vectors(dir){
  if(ip.read_register(60)!=0x53534e32)throw std::runtime_error("V2 ABI mismatch");
 }
 bool busy(){return ip.read_register(28)&1;}
 uint64_t read64(unsigned p){return uint64_t(ip.read_register(p))|(uint64_t(ip.read_register(p+4))<<32);}
 void write64(unsigned p,uint64_t x){ip.write_register(p,uint32_t(x));ip.write_register(p+4,uint32_t(x>>32));}
 void wait_done(){auto t=Clock::now();while(busy()){if(Clock::now()-t>std::chrono::seconds(10))throw std::runtime_error("Drain timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));}}
 void stop(){if(busy()){ip.write_register(0,2);wait_done();}}
 void load(unsigned r,const std::vector<uint32_t>& seed){
  if(busy()||r>=ip.read_register(72)||seed.size()!=12)throw std::runtime_error("Seed load configuration invalid");
  for(unsigned w=0;w<12;w++)ip.write_register(0x1000+r*256+w*4,seed[w]);
 }
 void run(unsigned mode,unsigned on,unsigned r,unsigned interval,uint64_t limit,bool allow_zero=false){
  if(busy()||mode>2||on>1||!r||r>ip.read_register(72)||!interval)throw std::runtime_error("Run configuration invalid");
  ip.write_register(0,8);ip.write_register(4,mode);ip.write_register(8,r);ip.write_register(12,on|(allow_zero?2:0));
  write64(16,limit);ip.write_register(24,interval);ip.write_register(0,1);
  if(ip.read_register(28)&4)throw std::runtime_error("Hardware rejected run");
 }
 void snapshot(const std::string& action){
  uint64_t before=now_ns();ip.write_register(0,4);uint64_t after=now_ns();
  uint64_t events=read64(32),cycles=read64(40),accepted=read64(64),batches=read64(48);
  unsigned status=ip.read_register(28),on=ip.read_register(12)&1;
  if(status&4)throw std::runtime_error("Hardware error/valid mismatch flag");
  std::cout<<"{\"action\":\""<<action<<"\",\"before_ns\":"<<before<<",\"after_ns\":"<<after<<",\"compute\":"<<(on?"true":"false")
   <<",\"events\":"<<events<<",\"completed\":"<<(on?events:0)<<",\"baseline_events\":"<<(on?0:events)<<",\"accepted\":"<<accepted
   <<",\"batches\":"<<batches<<",\"cycles\":"<<cycles<<",\"status\":"<<status<<",\"first_result_cycle\":"<<ip.read_register(56);
  if(action=="stop"){
   unsigned replicas=ip.read_register(8);
   std::cout<<",\"source_states\":[";
   for(unsigned r=0;r<replicas;r++){if(r)std::cout<<',';std::cout<<'[';for(unsigned w=0;w<12;w++){if(w)std::cout<<',';std::cout<<ip.read_register(0x1000+r*256+w*4);}std::cout<<']';}
   std::cout<<"],\"input_records\":[";
   for(unsigned r=0;r<replicas;r++){if(r)std::cout<<',';std::cout<<'[';for(unsigned w=0;w<32;w++){if(w)std::cout<<',';std::cout<<ip.read_register(0x4000+r*256+w*4);}std::cout<<']';}
   std::cout<<"],\"results\":[";for(unsigned r=0;r<replicas;r++){if(r)std::cout<<',';std::cout<<ip.read_register(0x800+r*4);}std::cout<<']';
  }
  std::cout<<"}"<<std::endl;
 }
 std::vector<Row> cases(const std::string& name){
  std::ifstream f(vectors+"/"+name);if(!f)throw std::runtime_error("Reference file missing");
  std::vector<Row> rows;Row r;std::string seed,state,record,result;
  while(f>>r.mode>>r.on>>r.n>>r.rep>>seed>>state>>record>>result){
   r.seed=hex_words(seed,12);r.state=hex_words(state,12);r.record=hex_words(record,32);r.result=std::stoul(result,nullptr,16);rows.push_back(r);
  }
  if(rows.empty()||!f.eof())throw std::runtime_error("Reference file malformed");
  return rows;
 }
 void check(unsigned replicas,bool smoke=false){
  if(replicas<1||replicas>ip.read_register(72))throw std::runtime_error("Invalid verification replica count");
  auto rows=cases(smoke?"smoke.txt":"controller.txt");unsigned checked=0,off_checked=0;
  unsigned width=smoke?1:16;
  for(size_t i=0;i<rows.size();i+=width){
   const auto& first=rows[i];if(ip.read_register(76)<3 && first.mode!=ip.read_register(76))continue;std::vector<uint32_t> previous;
   ip.write_register(4,first.mode);
   for(unsigned r=0;r<replicas;r++){
    const auto& row=rows.at(i+r);if(row.rep!=r)throw std::runtime_error("Reference replica order");
    load(r,row.seed);previous.push_back(ip.read_register(0x800+r*4));
   }
   run(first.mode,first.on,replicas,first.n==7?3:1,first.n,true);wait_done();ip.write_register(0,4);
   if(ip.read_register(28)&4)throw std::runtime_error("Verification hardware error");
   if(read64(32)!=uint64_t(first.n)*replicas||read64(64)!=uint64_t(first.n)*replicas)throw std::runtime_error("Verification event count mismatch");
   for(unsigned r=0;r<replicas;r++){
    const auto& row=rows[i+r];
    for(unsigned w=0;w<12;w++)if(ip.read_register(0x1000+r*256+w*4)!=row.state[w])throw std::runtime_error("Source state does not match independent oracle");
    for(unsigned w=0;w<32;w++)if(ip.read_register(0x4000+r*256+w*4)!=row.record[w])throw std::runtime_error("ON/OFF encoded input bank mismatch");
    auto actual=ip.read_register(0x800+r*4);
    if(first.on && actual!=row.result){std::ostringstream s;s<<"Dot mismatch format="<<row.mode<<" replica="<<r<<" actual="<<std::hex<<actual<<" expected="<<row.result;throw std::runtime_error(s.str());}
    if(!first.on && actual!=previous[r])throw std::runtime_error("OFF changed arithmetic output");
    checked+=bool(first.on);off_checked+=!first.on;
   }
  }
  std::cout<<"{\"action\":\""<<(smoke?"smoke":"check")<<"\",\"replicas\":"<<replicas<<",\"outputs_checked\":"<<checked<<",\"off_holds_checked\":"<<off_checked<<",\"input_and_state_checked\":true,\"passed\":true}"<<std::endl;
 }
 std::vector<uint32_t> power_run(unsigned mode,unsigned on,unsigned replicas,unsigned seedset,unsigned interval,uint64_t limit){
  if(seedset>=5)throw std::runtime_error("Invalid power seed set");
  std::ifstream f(vectors+"/power_seeds.bin",std::ios::binary);if(!f)throw std::runtime_error("Power seeds missing");
  f.seekg(seedset*16*48);
  for(unsigned r=0;r<replicas;r++){
   std::vector<uint32_t> words(12);for(auto& x:words){unsigned char b[4];f.read(reinterpret_cast<char*>(b),4);if(!f)throw std::runtime_error("Truncated seeds");x=uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);}
   load(r,words);
  }
  ip.write_register(4,mode);
  std::vector<uint32_t> previous;
  for(unsigned r=0;r<replicas;r++)previous.push_back(ip.read_register(0x800+r*4));
  run(mode,on,replicas,interval,limit);
  return previous;
 }
};
}
int main(int argc,char** argv){
 if(argc!=4){std::cerr<<"DEVICE IMAGE VECTORS required\n";return 2;}
 try{
  Controller c(argv[1],argv[2],argv[3]);std::cout<<"{\"action\":\"ready\",\"abi\":\"SSN2\",\"max_replicas\":"<<c.ip.read_register(72)<<",\"format_id\":"<<c.ip.read_register(76)<<"}"<<std::endl;
  try{
   std::string line;
   while(std::getline(std::cin,line)){
    std::istringstream f(line);std::string cmd;f>>cmd;
    if(cmd=="smoke")c.check(1,true);
    else if(cmd=="check"){unsigned r;f>>r;if(!f)throw std::runtime_error("check replicas required");c.check(r);}
    else if(cmd=="run"){
     unsigned mode,on,r,seed,interval;uint64_t limit;f>>mode>>on>>r>>seed>>interval>>limit;if(!f)throw std::runtime_error("run mode on replicas seed interval limit required");
     auto previous=c.power_run(mode,on,r,seed,interval,limit);
     std::cout<<"{\"action\":\"run\",\"initial_results\":[";
     for(unsigned k=0;k<previous.size();k++){if(k)std::cout<<',';std::cout<<previous[k];}
     std::cout<<"]}"<<std::endl;
    }
    else if(cmd=="snapshot")c.snapshot("snapshot");
    else if(cmd=="stop"){c.stop();c.snapshot("stop");}
    else if(cmd=="quit"){c.stop();std::cout<<"{\"action\":\"quit\"}"<<std::endl;return 0;}
    else throw std::runtime_error("Unknown command");
   }
   c.stop();
  }catch(...){c.stop();throw;}
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
