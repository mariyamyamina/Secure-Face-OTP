import { Router, type IRouter } from "express";
import healthRouter from "./health";
import authRouter from "./auth";
import otpRouter from "./otp";
import adminRouter from "./admin";
import livenessRouter from "./liveness";

const router: IRouter = Router();

router.use(healthRouter);
router.use(livenessRouter);
router.use(authRouter);
router.use(otpRouter);
router.use(adminRouter);

export default router;
