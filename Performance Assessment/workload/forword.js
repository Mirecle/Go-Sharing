'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class MyWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }
    
   
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
	for (let i=0;i<this.roundArguments.assets;i++){
		const HashP = `${this.workerIndex}_${i}`;
		console.log(`Hash of picture:${HashP}`)
		const request = {
			contractId: 'asn2',
            	contractFunction: 'UploadPhoto',
            	invokerIdentity: 'Admin@org2.example.com',
            	contractArguments: ['OA','OSN1','OSNpk','PointerPA','PointerPolivyP',HashP],
            	readOnly: false
		};	
		await this.sutAdapter.sendRequests(request);
	}

    }
    
    async submitTransaction() {
	  const random=Math.floor(Math.random()*this.roundArguments.assets);
        console.log(`forwordPhoto ${this.workerIndex}_${random}`);
        const myArgs = {
            contractId: 'asn2',
            contractFunction: 'ForwordPhoto',
            invokerIdentity: 'Admin@org2.example.com',
            contractArguments: ['v2','OSN3','v1+v2+v3+v4+v5,v1+v2+v3+v4,v1+v2+v3,v1','OA','OSN1','OSN1pk','OSN3pk',`${this.workerIndex}_${random}`],
            readOnly: false
        };

        await this.sutAdapter.sendRequests(myArgs);
        
    }
    
    async cleanupWorkloadModule() {
       
    }
    

    

  
}



function createWorkloadModule() {
    return new MyWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;

