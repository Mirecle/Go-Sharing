'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class MyWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }


    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {

       await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
	this.workerIndex = workerIndex;
        this.totalWorkers = totalWorkers;
        this.roundIndex = roundIndex;
        this.roundArguments = roundArguments;
        this.sutAdapter = sutAdapter;
        this.sutContext = sutContext;

    }

    async submitTransaction() {


        const hash = new Date().getTime().toString();
	   const padding= this.workerIndex.toString();
	   const random=Math.floor(Math.random()*this.roundArguments.assets);
        console.log(`Worker ${this.workerIndex}:sutAdapter"`+hash+padding+random);
        const myArgs = {
            contractId: 'asn',
            contractFunction: 'DeletePhoto',
            invokerIdentity: 'Admin@org1.example.com',
            contractArguments: ['OA','OSN1',hash+padding+random],
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

