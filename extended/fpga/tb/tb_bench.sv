`timescale 1ns/1ps
module tb_bench #(parameter integer TEST_REPLICAS=2,parameter integer TEST_FORMAT=-1);
  reg clk=0;always #5 clk=~clk;
  reg resetn=0;
  reg [15:0] awaddr=0,araddr=0;
  reg awvalid=0,wvalid=0,bready=0,arvalid=0,rready=0;
  reg [31:0] wdata=0;
  reg [3:0] wstrb=15;
  wire awready,wready,bvalid,arready,rvalid;
  wire [1:0] bresp,rresp;
  wire [31:0] rdata;
  ssnf_bench #(.MAX_REPLICAS(TEST_REPLICAS),.STATIC_FORMAT(TEST_FORMAT)) dut(clk,resetn,awaddr,awvalid,awready,wdata,wstrb,wvalid,wready,
    bresp,bvalid,bready,araddr,arvalid,arready,rdata,rresp,rvalid,rready);

 reg [383:0] seeds[0:15],next_states[0:15];
 reg [1023:0] records[0:15];
 reg [31:0] expected[0:15],before_results[0:15];
 reg [383:0] sv,nv;
 reg [1023:0] rec;
 reg [31:0] y,value,completed,accepted;
 integer f,rc,mode,compute,batches,rep,other_m,other_c,other_b,groups=0,polls;
 string filename;
 for(genvar r=0;r<TEST_REPLICAS;r=r+1) begin: off_checks
  always @(posedge clk) if(resetn&&dut.busy&&!dut.cfg_flags[0]) begin
   if(dut.replica_g[r].ssnf_record!==0 || dut.replica_g[r].mx_record!==0 || dut.replica_g[r].bf_record!==0)
    $fatal(1,"OFF arithmetic inputs not isolated");
   if(dut.replica_g[r].computing!==0)$fatal(1,"OFF arithmetic enable active");
  end
 end
  task automatic write_reg(input [15:0] addr,input [31:0] data,input [1:0] response);
    begin
      if(addr[2]) begin // W may arrive before AW.
        @(negedge clk);wdata=data;wvalid=1;
        do @(posedge clk); while(!wready);
        @(negedge clk);wvalid=0;
        @(negedge clk);awaddr=addr;awvalid=1;
        do @(posedge clk); while(!awready);
        @(negedge clk);awvalid=0;
      end else begin
        @(negedge clk);awaddr=addr;awvalid=1;
        do @(posedge clk); while(!awready);
        @(negedge clk);awvalid=0;
        @(negedge clk);wdata=data;wvalid=1;
        do @(posedge clk); while(!wready);
        @(negedge clk);wvalid=0;
      end
      repeat(2) @(negedge clk); // B backpressure must retain response.
      bready=1;
      do @(posedge clk); while(!bvalid);
      if(bresp!==response) $fatal(1,"AXI write response addr=%h got=%h expected=%h",addr,bresp,response);
      @(negedge clk);bready=0;
    end
  endtask
  task automatic read_reg(input [15:0] addr,output [31:0] data);
    begin
      @(negedge clk);araddr=addr;arvalid=1;
      do @(posedge clk); while(!arready);
      @(negedge clk);arvalid=0;
      repeat(2) @(negedge clk);
      rready=1;
      do @(posedge clk); while(!rvalid);
      if(rresp!==0) $fatal(1,"AXI read response addr=%h",addr);
      data=rdata;
      @(negedge clk);rready=0;
    end
  endtask
  task automatic wait_done;
    begin
      polls=0;value=1;
      while(value[0]) begin
        read_reg(16'h1c,value);polls=polls+1;
        if(polls>1000) $fatal(1,"Benchmark failed to drain");
      end
      if(!value[1]) $fatal(1,"Done bit missing");
    end
  endtask

 initial begin
  if(!$value$plusargs("VECTORS=%s",filename))$fatal(1,"vectors needed");
  f=$fopen(filename,"r");if(!f)$fatal(1,"vectors missing");
  repeat(3) @(negedge clk);resetn=1;
  read_reg(60,value);if(value!==32'h53534e32)$fatal(1,"ABI mismatch");
  if(TEST_FORMAT>=0)write_reg(4,TEST_FORMAT,0);
  write_reg(0,1,0);read_reg(28,value);if(!value[2]||value[0])$fatal(1,"Zero seed not rejected");
  write_reg(0,8,0);
  while(!$feof(f)) begin
   rc=$fscanf(f,"%d %d %d %d %h %h %h %h\n",mode,compute,batches,rep,sv,nv,rec,y);
   if(rc==8) begin
    if(rep!=0)$fatal(1,"case order");
    seeds[0]=sv;next_states[0]=nv;records[0]=rec;expected[0]=y;
    for(integer r=1;r<16;r=r+1) begin
     rc=$fscanf(f,"%d %d %d %d %h %h %h %h\n",other_m,other_c,other_b,rep,sv,nv,rec,y);
     if(rc!=8||rep!=r||other_m!=mode||other_c!=compute||other_b!=batches)$fatal(1,"case shape");
     seeds[r]=sv;next_states[r]=nv;records[r]=rec;expected[r]=y;
    end
    if(TEST_FORMAT<0||mode==TEST_FORMAT) begin
    for(integer r=0;r<TEST_REPLICAS;r=r+1)
     for(integer w=0;w<12;w=w+1) begin
      if(w%5==0) begin
       wstrb=3;write_reg(16'h1000+r*256+w*4,seeds[r][w*32+:32],0);
       wstrb=12;write_reg(16'h1000+r*256+w*4,seeds[r][w*32+:32],0);wstrb=15;
      end else write_reg(16'h1000+r*256+w*4,seeds[r][w*32+:32],0);
     end
    write_reg(4,mode,0);write_reg(8,TEST_REPLICAS,0);write_reg(12,compute,0);
    write_reg(16,batches,0);write_reg(20,0,0);write_reg(24,(batches==7)?3:1,0);
    for(integer r=0;r<TEST_REPLICAS;r=r+1) read_reg(16'h800+r*4,before_results[r]);
    write_reg(0,1,0);wait_done();write_reg(0,4,0);
    read_reg(28,value);if(value[2])$fatal(1,"Controller validity/config error mode=%0d compute=%0d",mode,compute);
    read_reg(32,completed);read_reg(64,accepted);
    if(completed!=batches*TEST_REPLICAS||accepted!=completed)$fatal(1,"Count mismatch");
    for(integer r=0;r<TEST_REPLICAS;r=r+1) begin
     for(integer w=0;w<12;w=w+1) begin
      read_reg(16'h1000+r*256+w*4,value);
      if(value!==next_states[r][w*32+:32])$fatal(1,"Source progress mismatch mode=%0d compute=%0d r=%0d",mode,compute,r);
     end
     for(integer w=0;w<32;w=w+1) begin
      read_reg(16'h4000+r*256+w*4,value);
      if(value!==records[r][w*32+:32])$fatal(1,"Input record mismatch");
     end
     read_reg(16'h800+r*4,value);
     if(compute && value!==expected[r])$fatal(1,"Dot result mismatch mode=%0d r=%0d actual=%h expected=%h",mode,r,value,expected[r]);
     if(!compute && value!==before_results[r])$fatal(1,"OFF changed arithmetic output");
    end
    $display("PASS AXI mode=%0d compute=%0d R=%0d batches=%0d events=%0d",mode,compute,TEST_REPLICAS,batches,completed);
    groups=groups+1;
    end
   end
  end
  write_reg(12,1,0);write_reg(16,0,0);write_reg(0,1,0);
  repeat(100) @(negedge clk);
  write_reg(16'h1000,32'h12345678,2);
  write_reg(0,2,0);wait_done();write_reg(0,4,0);
  read_reg(32,completed);read_reg(64,accepted);
  if(completed!=accepted||completed==0)$fatal(1,"Continuous drain mismatch");
  read_reg(28,value);if(!value[2])$fatal(1,"busy write guard missing");
  write_reg(0,8,0);write_reg(8,TEST_REPLICAS+1,0);write_reg(0,1,0);
  read_reg(28,value);if(!value[2]||value[0])$fatal(1,"Replica limit guard missing");
  write_reg(0,8,0);write_reg(8,1,0);write_reg(0,1,0);
  repeat(3) @(negedge clk);resetn=0;repeat(3)@(negedge clk);resetn=1;
  repeat(30)@(negedge clk);write_reg(0,4,0);read_reg(32,value);
  if(value!==0)$fatal(1,"Reset did not clear counters");
  $display("PASS V2 CONTROLLER %0d groups, ON/OFF exact input identity, outputs, events, AXI ordering/backpressure/partial writes, stop and guards",groups);$finish;
 end
 initial begin #1000000000;$fatal(1,"timeout");end

endmodule
