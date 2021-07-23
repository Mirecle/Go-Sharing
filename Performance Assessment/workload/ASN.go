/**
 * @Author: ChZheng
 * @Date: 2021-01-07 17:17:27
 * @LastEditTime: 2021-03-13 20:10:05
 * @LastEditors: ChZheng
 * @Description:新版 ASN 链码
 * @FilePath: /ASN-blockchain/Users/apple/go/src/github.com/hyperledger/fabric-samples/chaincode/ASN/ASN.go
 */
//TODO:命名规范
//TODO:白名单黑名单
//TODO:queryState
package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/peer"
)

// ASNChaincode implementation
type ASNChaincode struct{}

func main() {
	err := shim.Start(new(ASNChaincode))
	if err != nil {
		fmt.Printf("Error starting Simple chaincode: %s", err)
	}
}

// Init initializes the chaincode
func (t *ASNChaincode) Init(stub shim.ChaincodeStubInterface) peer.Response {
	policyPointer := "PointerPolivyP"
	//"AGp,DGp,RGp,RUGp"
	policyP := "v1+v2+v3+v4+v5,v1+v2+v3+v4,v1+v2+v3,v1+v2"

	stub.PutState(policyPointer, []byte(policyP))
	stub.PutState("Pid", []byte("1"))
	stub.PutState("HashPA", []byte("1"))
	stub.PutState("IDARL", []byte("1"))
	stub.PutState("PidChainID", []byte("1"))

	return shim.Success(nil)
}

// Invoke functions
func (t *ASNChaincode) Invoke(stub shim.ChaincodeStubInterface) peer.Response {
	fun, args := stub.GetFunctionAndParameters()
	if fun == "UploadPhoto" {
		return uploadPhoto(stub, args)
	} else if fun == "VisitPhoto" {
		return visitOrDownloadPhoto(stub, args, "Visiting")
	} else if fun == "DownloadPhoto" {
		return visitOrDownloadPhoto(stub, args, "Downloading")
	} else if fun == "ForwordPhoto" {
		return forwordOrReuploadPhoto(stub, args, "Forwording")
	} else if fun == "ReuploadPhoto" {
		return forwordOrReuploadPhoto(stub, args, "Reuploading")
	} else if fun == "DeletePhoto" {
		return deletePhoto(stub, args)
	}
	return shim.Error("Commend is not defined")
}
func uploadPhoto(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	UserID := args[0]
	OSNID := args[1]
	PointerPA := args[2]
	PointerPolivyP := args[3]
	HashPA := args[4]

	checkExistencePhoto(stub, HashPA, UserID, OSNID)
	SMCaddr := createSMC(stub, HashPA, UserID, OSNID, PointerPA, PointerPolivyP)
	PMCC := UserID + "," + OSNID + "," + SMCaddr
	stub.PutState(HashPA, []byte(PMCC))

	return shim.Success(nil)

}

func visitOrDownloadPhoto(stub shim.ChaincodeStubInterface, args []string, OperationT string) peer.Response {
	VB := args[0]
	OSN2 := args[1]
	OA := args[2]
	OSN1 := args[3]
	HashPA := args[4]

	checkAccPro(stub, HashPA, OA, OSN1)

	stub.GetState("Pid")
	PolicyP, _ := stub.GetState("policyPointer") //其实是存在数据库中的
	result := checkPolicy(string(PolicyP), OperationT, VB)

	addAccessRecord(stub, HashPA, VB, OSN2, OperationT, result)

	return shim.Success(nil)
}

func forwordOrReuploadPhoto(stub shim.ChaincodeStubInterface, args []string, OperationT string) peer.Response {
	VC := args[0]
	OSN3 := args[1]
	VCpolicyP := args[2]
	OA := args[3]
	OSN1 := args[4]
	HashPA := args[5]

	checkAccPro(stub, HashPA, OA, OSN1)

	stub.GetState("Pid")
	VApolicyP, _ := stub.GetState("policyPointer")
	result := checkPolicy(string(VApolicyP), OperationT, VC) //未加入组织权限
	SMCaddr := "S" + HashPA + OA + OSN1
	createPolicy(string(VApolicyP), VCpolicyP)
	addAccessRecord(stub, HashPA, VC, OSN3, OperationT, result)
	addPropagationChain(stub, HashPA, SMCaddr, VC, OSN3, "PointerPANew", "PointerPolicyPNew")
	return shim.Success(nil)

}

func deletePhoto(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	UserID := args[0]
	OSNID := args[1]
	HashPA := args[2]
	OperationT := "Delete"
	result := "adopt"

	checkAccPro(stub, HashPA, UserID, OSNID)
	SMCaddr := "S" + HashPA + UserID + OSNID
	PidChainID := "CP" + SMCaddr
	delNode(stub, PidChainID)

	addAccessRecord(stub, HashPA, UserID, OSNID, OperationT, result)

	return shim.Success(nil)
}

func checkExistencePhoto(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNID string) bool {
	stub.GetState("HashPA")

	ProChain, _ := stub.GetState("Pid")
	ProChainStr := getStrings(string(ProChain), '+')
	if len(ProChainStr) != 5 {
		return false
	}
	Status := ProChainStr[4]
	return Status == "Active" //it is bool
}

func addAccessRecord(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNId string, OperationT string, result string) (string, string) {
	IDARL := "I" + HashPA
	var Pid string
	Timestamp := time.Now().Format("20060102150405")
	if OperationT == "Visiting" || OperationT == "Downloading" || OperationT == "Delete" || result == "refuse" {
		Pid = "0"
	} else {
		Pid = "PS" + HashPA + UserID + OSNId
	}
	var AccRecList string
	if OperationT == "Uploading" {
		AccRecList = UserID + "+" + OSNId + "+" + OperationT + "+" + result + "+" + Timestamp + "+" + Pid
	} else {
		stub.GetState("IDARL")
		AccRecList = UserID + "+" + OSNId + "+" + OperationT + "+" + result + "+" + Timestamp + "+" + Pid
	}
	err := stub.PutState(IDARL, []byte(AccRecList))
	if err != nil {
		IDARL = err.Error()
	}
	return IDARL, AccRecList
}

func addPropagationChain(stub shim.ChaincodeStubInterface, HashPA string, SMCaddr string, UserID string, OSNID string, PhotoAddress string, PolicyAddress string) string {
	PidChainID := "CP" + SMCaddr
	Pid := "PS" + HashPA + UserID + OSNID
	FrontID := "P" + SMCaddr //第一次上传指向自己，二次上传或转发指向前一个节点
	// when test zhushi this
	stub.GetState("Pid")

	var ProChain, Status, PidChain string
	Status = "Active"
	stub.GetState("PidChainID") //同源图片在一条pid 链上

	PidChain = Pid

	ProChain = Pid + "+" + PhotoAddress + "+" + PolicyAddress + "+" + FrontID + "+" + Status
	err2 := stub.PutState(PidChainID, []byte(PidChain)) //Pid 链
	if err2 != nil {
		return "PidChainError" + err2.Error()
	}
	err3 := stub.PutState(Pid, []byte(ProChain)) //传播链
	if err3 != nil {
		return "ProChainError" + err3.Error()
	}
	return PidChainID
}
func createSMC(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNID string, PhotoAddress string, PolicyAddress string) string {
	SMCaddr := "S" + HashPA + UserID + OSNID
	OperationT := "Uploading"
	Result := " "
	IDARL, _ := addAccessRecord(stub, HashPA, UserID, OSNID, OperationT, Result)
	PidChainID := addPropagationChain(stub, HashPA, SMCaddr, UserID, OSNID, PhotoAddress, PolicyAddress)
	SMCC := IDARL + "," + PidChainID

	err := stub.PutState(SMCaddr, []byte(SMCC)) //SMC
	if err != nil {
		return "SMCaddrError" + err.Error()
	}
	return SMCaddr
}

func getValues(stub shim.ChaincodeStubInterface, Key string) []string {
	value, _ := stub.GetState(Key)
	if value == nil {
		return nil
	}
	valueStr := getStrings(string(value), ',')
	return valueStr
}

func getStrings(str string, sym rune) []string {
	f := func(c rune) bool {
		return c == sym
	}
	strs := strings.FieldsFunc(str, f)
	return strs
}

func checkAccPro(stub shim.ChaincodeStubInterface, HashPA string, UserID string, OSNId string) string {
	var err string
	err = "nil"
	if !checkExistencePhoto(stub, HashPA, UserID, OSNId) {
		err = "There is no picture in Picture pool matched " + HashPA
		return err
	}

	AccRecList, _ := stub.GetState("IDARL")
	if AccRecList == nil {
		err = OSNId + "." + UserID + " is an illegal Photo owner"
		return err
	}
	ProChain, _ := stub.GetState("Pid")
	if ProChain == nil {
		err = "err ProChain1"
		return err

	}
	ProChainStr := getStrings(string(ProChain), '+')
	if len(ProChainStr) != 5 {
		err = "err ProChain2"
		return err
	}
	Status := ProChainStr[4]
	if Status != "Active" {
		err = "The propagation state of " + OSNId + "." + UserID + " is not active"
		return err
	}
	return err
}

func checkPolicy(PolicyP string, OperationT string, PhotoVisitor string) string {
	PolicyPstr := getStrings(PolicyP, ',')
	if len(PolicyPstr) == 0 {
		return "refuse"
	}
	AGpstr := getStrings(PolicyPstr[0], '+')
	DGpstr := getStrings(PolicyPstr[1], '+')
	RGpstr := getStrings(PolicyPstr[2], '+')
	RUGpstr := getStrings(PolicyPstr[3], '+')

	if len(AGpstr) == 0 || len(DGpstr) == 0 || len(RGpstr) == 0 || len(RUGpstr) == 0 {
		return "refuse1"
	}
	if OperationT == "Visiting" {
		for _, v := range AGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}
		}
	} else if OperationT == "Downloading" {
		for _, v := range DGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}

		}
	} else if OperationT == "Forwording" {
		for _, v := range RGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}
		}
	} else if OperationT == "Reuploading" {
		for _, v := range RUGpstr {
			if v == PhotoVisitor {
				return "adopt"
			}

		}
	}

	return "refuse2"
}

func createPolicy(PolicyP1 string, PolicyP2 string) string {
	PolicyPstr1 := getStrings(PolicyP1, ',')
	if len(PolicyPstr1) == 0 {
		return "refuse"
	}
	AGpstr1 := getStrings(PolicyPstr1[0], '+')
	DGpstr1 := getStrings(PolicyPstr1[1], '+')
	RGpstr1 := getStrings(PolicyPstr1[2], '+')
	RUGpstr1 := getStrings(PolicyPstr1[3], '+')

	if len(AGpstr1) == 0 || len(DGpstr1) == 0 || len(RGpstr1) == 0 || len(RUGpstr1) == 0 {
		return "refuse"
	}
	PolicyPstr2 := getStrings(PolicyP2, ',')
	if len(PolicyPstr2) == 0 {
		return "refuse"
	}
	AGpstr2 := getStrings(PolicyPstr2[0], '+')
	DGpstr2 := getStrings(PolicyPstr2[1], '+')
	RGpstr2 := getStrings(PolicyPstr2[2], '+')
	RUGpstr2 := getStrings(PolicyPstr2[3], '+')

	if len(AGpstr2) == 0 || len(DGpstr2) == 0 || len(RGpstr2) == 0 || len(RUGpstr2) == 0 {
		return "refuse"
	}
	AGp3 := intersection(AGpstr1, AGpstr2)
	DGp3 := intersection(DGpstr1, DGpstr2)
	RGp3 := intersection(RGpstr1, RGpstr2)
	RUGp3 := intersection(RUGpstr1, RUGpstr2)
	PolicyP3 := AGp3 + "," + DGp3 + "," + RGp3 + "," + RUGp3
	return PolicyP3
}

func intersection(a []string, b []string) (inter string) {
	// interacting on the smallest list first can potentailly be faster...but not by much,worse case is the same
	low, high := a, b
	if len(a) > len(b) {
		low = b
		high = a
	}

	done := false
	for i, l := range low {
		for j, h := range high {
			f1 := i + 1
			f2 := j + 1
			if l == h {
				inter = inter + "+" + h
				if f1 < len(low) && f2 < len(high) {

					if low[f1] != high[f2] {
						done = true
					}
				}

				high = high[:j+copy(high[j:], high[j+1:])]
				break
			}
		}
		if done {
			break
		}
	}
	return
}

func delNode(stub shim.ChaincodeStubInterface, PidChainID string) bool {
	PidStr := getValues(stub, PidChainID)
	if PidStr == nil {
		return false
	}
	for _, Pid := range PidStr {
		ProC, _ := stub.GetState(Pid)

		getStrings(string(ProC), '+')

		err2 := stub.PutState(Pid, []byte("ProChain")) //传播链
		if err2 != nil {
			return false
		}
		if "C"+Pid == PidChainID {
			continue
		}
	}
	return true
}

