"""六个修复点及配套管线的语义测试（独立于有结构缺陷的原始契约测试）。"""

import math

import pytest

import graphics.motion_blur_shutter as mod


# --- 修复 1：快门与帧时间解耦 ---------------------------------------------

class TestShutterDecoupling:
    def test_window_length_independent_of_frame_time(self):
        # 同样的曝光时间，60Hz 与掉帧后的 30Hz 帧间隔，窗口长度必须一致
        steady = mod.shutter_time({"exposure": 1 / 120, "frame_time": 1 / 60})
        dropped = mod.shutter_time({"exposure": 1 / 120, "frame_time": 1 / 30})
        assert steady["exposure"] == pytest.approx(dropped["exposure"])
        assert steady["requested_exposure"] == pytest.approx(dropped["requested_exposure"])

    def test_centered_shutter_window(self):
        ft = 1 / 60
        out = mod.shutter_time({"exposure": 1 / 120, "frame_time": ft,
                                "shutter_offset": -0.5})
        # 居中：open/close 关于 frame_time/2 对称，窗口不越界
        assert out["center"] == pytest.approx(ft / 2)
        assert out["open"] == pytest.approx(ft / 2 - (1 / 120) / 2)
        assert out["close"] == pytest.approx(ft / 2 + (1 / 120) / 2)
        assert out["overrun"] is False

    def test_overrun_flagged_not_resized(self):
        out = mod.shutter_time({"exposure": 1 / 30, "frame_time": 1 / 60,
                                "shutter_offset": -0.5})
        assert out["overrun"] is True
        # 不截断时曝光长度保持请求值
        assert out["exposure"] == pytest.approx(1 / 30)

    def test_clamp_shortens_only_when_requested(self):
        out = mod.shutter_time({"exposure": 1 / 30, "frame_time": 1 / 60,
                                "clamp_to_frame": True})
        assert out["clamped"] is True
        assert out["open"] == pytest.approx(0.0, abs=1e-12)
        assert out["close"] == pytest.approx(1 / 60)

    def test_invalid_inputs_fall_back(self):
        out = mod.shutter_time({"exposure": 0, "frame_time": -1})
        assert out["exposure"] > 0 and out["frame_time"] > 0
        assert out["warnings"]


# --- 修复 1b：掉帧在速度域补偿，拖影不突变 ---------------------------------

class TestFrameDrop:
    def test_dropped_count(self):
        out = mod.frame_drop({"frame_time": 2 / 60, "expected_frame_time": 1 / 60})
        assert out["dropped_frames"] == 1
        assert out["velocity_scale"] == pytest.approx(0.5)

    def test_exposure_unchanged_through_drop(self):
        out = mod.frame_drop({"frame_time": 3 / 60, "exposure": 1 / 120})
        assert out["exposure_unchanged"] is True
        assert out["shutter"]["exposure"] == pytest.approx(1 / 120)

    def test_no_drop_steady_state(self):
        out = mod.frame_drop({"frame_time": 1 / 60, "expected_frame_time": 1 / 60})
        assert out["dropped_frames"] == 0
        assert out["velocity_scale"] == pytest.approx(1.0)


# --- 修复 2：旋转与平移合成单一矢量 ----------------------------------------

class TestVectorCompose:
    def test_translation_plus_angle(self):
        out = mod.vector_compose({
            "translation": (10.0, 0.0),
            "rotation": 0.0,
            "focal_px": 500.0,
        })
        # 零角度旋转贡献零像素位移
        assert out["vector"] == pytest.approx((10.0, 0.0))

    def test_rotation_px_uses_pinhole_model(self):
        angle = 0.01
        out = mod.vector_compose({
            "translation": (0.0, 0.0),
            "rotation": angle,
            "focal_px": 500.0,
        })
        assert out["rotation_px"][0] == pytest.approx(500.0 * math.tan(angle))
        assert out["vector"] == pytest.approx(out["rotation_px"])

    def test_single_direction_not_two_vectors(self):
        out = mod.vector_compose({
            "translation": (3.0, 4.0),
            "rotation": {"angle": 0.0, "axis": "x"},
        })
        vx, vy = out["vector"]
        assert out["direction"] == pytest.approx(math.atan2(vy, vx))
        assert out["magnitude"] == pytest.approx(math.hypot(vx, vy))

    def test_rotation_axis_y(self):
        out = mod.vector_compose({
            "rotation": {"angle": 0.02, "axis": "y"},
            "focal_px": 100.0,
        })
        assert out["rotation_px"][0] == pytest.approx(0.0)
        assert out["rotation_px"][1] == pytest.approx(100.0 * math.tan(0.02))

    def test_rotation_path_alignment(self):
        out = mod.rotation_path({"angle_start": 0.0, "angle_end": 0.04, "steps": 4})
        samples = out["samples"]
        assert [s["t"] for s in samples] == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
        assert samples[0]["angle"] == pytest.approx(0.0)
        assert samples[-1]["angle"] == pytest.approx(0.04)
        # 等步长角度
        deltas = [samples[i + 1]["angle"] - samples[i]["angle"] for i in range(4)]
        assert deltas == pytest.approx([0.01] * 4)


# --- 修复 3：历史分离 ------------------------------------------------------

class TestHistorySplit:
    def test_independent_storage(self):
        out = mod.history_split({"entries": [1, 2, 3], "size": 4})
        blur, aa = out["blur_history"], out["aa_history"]
        assert blur is not aa
        assert blur["entries"] is not aa["entries"]
        assert out["aliased"] is False and out["separate_storage"] is True

    def test_invalidating_one_side_keeps_other(self):
        out = mod.history_split({"entries": [{"frame": 7, "valid": False}, 8]})
        assert out["blur_history"]["valid_flags"] == [False, True]
        assert out["aa_history"]["valid_flags"] == [False, True]
        # 两个缓冲独立失效后结构仍可独立修改
        out["blur_history"]["invalidated"] = True
        assert out["aa_history"]["invalidated"] is False


# --- 修复 4：蒙皮后重算矢量 ------------------------------------------------

class TestSkinMotion:
    def test_vectors_recomputed_from_skinned_positions(self):
        out = mod.skin_motion({
            "vertices": [{"id": "a", "pos": (2.0, 0.0)},
                         {"id": "b", "pos": (0.0, 4.0)}],
            "prev_vertices": [{"id": "a", "pos": (0.0, 0.0)},
                              {"id": "b", "pos": (0.0, 0.0)}],
            "frame_time": 0.5,
        })
        vectors = {v["id"]: v["vector"] for v in out["vertices"]}
        assert vectors["a"] == pytest.approx((4.0, 0.0))
        assert vectors["b"] == pytest.approx((0.0, 8.0))
        assert out["updated_after_skinning"] is True

    def test_disoccluded_vertex_zero_vector(self):
        out = mod.skin_motion({
            "vertices": [{"id": "new", "pos": (1.0, 1.0)}],
            "prev_vertices": [],
        })
        assert out["vertices"][0]["disoccluded"] is True
        assert out["vertices"][0]["vector"] == pytest.approx((0.0, 0.0))
        assert out["disoccluded_count"] == 1

    def test_id_matching_order_independent(self):
        out = mod.skin_motion({
            "vertices": [{"id": "b", "pos": (0.0, 2.0)},
                         {"id": "a", "pos": (2.0, 0.0)}],
            "prev_vertices": [{"id": "a", "pos": (0.0, 0.0)},
                              {"id": "b", "pos": (0.0, 0.0)}],
            "frame_time": 1.0,
        })
        by_id = {v["id"]: v["vector"] for v in out["vertices"]}
        assert by_id["a"] == pytest.approx((2.0, 0.0))
        assert by_id["b"] == pytest.approx((0.0, 2.0))


# --- 修复 5：结果帧对齐 ----------------------------------------------------

class TestResultAlign:
    def test_apply_when_frame_matches_and_not_presented(self):
        out = mod.result_align({
            "source_frame_time": 1.0, "target_frame_time": 1.0,
            "presented": False, "result": {"bloom": 1},
        })
        assert out["status"] == "applied" and out["aligned"] is True
        assert out["reason"] == "ok"

    def test_discard_when_already_presented(self):
        out = mod.result_align({
            "source_frame_time": 1.0, "target_frame_time": 1.0,
            "presented": True,
        })
        assert out["status"] == "discarded"
        assert out["reason"] == "target_already_presented"

    def test_discard_on_frame_mismatch(self):
        out = mod.result_align({
            "source_frame_time": 1.0, "target_frame_time": 2.0,
            "presented": False,
        })
        assert out["status"] == "discarded" and out["reason"] == "frame_mismatch"

    def test_present_time_inference(self):
        late = mod.result_align({
            "source_frame_time": 1.0, "target_frame_time": 1.0,
            "present_time": 1.05, "arrival_time": 1.08,
        })
        assert late["aligned"] is False
        early = mod.result_align({
            "source_frame_time": 1.0, "target_frame_time": 1.0,
            "present_time": 1.05, "arrival_time": 1.02,
        })
        assert early["aligned"] is True


class TestAsyncPost:
    def test_queue_routing(self):
        out = mod.async_post({
            "now": 2.0,
            "jobs": [
                {"job_id": "j1", "ready": True, "source_frame_time": 1.0,
                 "target_frame_time": 1.0, "presented": False},
                {"job_id": "j2", "ready": True, "source_frame_time": 1.0,
                 "target_frame_time": 1.0, "presented": True},
                {"job_id": "j3", "ready": False, "source_frame_time": 1.0},
            ],
        })
        assert {j["job_id"] for j in out["applied"]} == {"j1"}
        assert {j["job_id"] for j in out["discarded"]} == {"j2"}
        assert {j["job_id"] for j in out["pending"]} == {"j3"}
        assert (out["applied_count"], out["discarded_count"], out["pending_count"]) == (1, 1, 1)

    def test_presented_frames_set(self):
        out = mod.async_post({
            "jobs": [{"job_id": "j", "source_frame_time": 5.0,
                      "target_frame_time": 5.0}],
            "presented_frames": [5.0],
        })
        assert out["discarded_count"] == 1


# --- 修复 6：拖影上限 ------------------------------------------------------

class TestTrailCap:
    def test_cap_preserves_direction(self):
        out = mod.trail_cap({"vectors": {"x": 30.0, "y": 40.0}, "max_trail_px": 25.0})
        capped = out["capped"]
        assert capped["magnitude"] == pytest.approx(25.0)
        assert capped["capped"] is True
        # 方向不变：归一化后与原方向一致
        assert capped["vector"][0] / 25.0 == pytest.approx(0.6)
        assert capped["vector"][1] / 25.0 == pytest.approx(0.8)

    def test_under_limit_untouched(self):
        out = mod.trail_cap({"vectors": {"x": 3.0, "y": 4.0}, "max_trail_px": 10.0})
        assert out["capped"]["capped"] is False
        assert out["capped"]["vector"] == pytest.approx((3.0, 4.0))
        assert out["capped"]["scale"] == pytest.approx(1.0)

    def test_scalar_shim(self):
        out = mod.trail_cap(1)
        assert out["capped"]["vector"] == pytest.approx((1.0, 0.0))

    def test_batch_and_count(self):
        out = mod.trail_cap({
            "vectors": [(30.0, 40.0), (1.0, 0.0), (0.0, 200.0)],
            "max_trail_px": 25.0,
        })
        assert out["capped_count"] == 2 and out["total_count"] == 3
        assert out["results"][1]["capped"] is False


# --- TAA 抖动与运动矢量隔离 ------------------------------------------------

class TestCameraJitter:
    def test_deterministic_halton_and_zero_blur_motion(self):
        a = mod.camera_jitter({"frame_index": 0})
        b = mod.camera_jitter({"frame_index": 0})
        assert a["projection_jitter"] == b["projection_jitter"]
        assert a["blur_motion"] == (0.0, 0.0)
        assert a["affects_motion_vectors"] is False
        # Halton(2) 第一项 = 1/2 -> 居中到 0
        assert a["projection_jitter"][0] == pytest.approx(0.0)

    def test_different_frames_different_jitter(self):
        j0 = mod.camera_jitter({"frame_index": 0})["projection_jitter"]
        j1 = mod.camera_jitter({"frame_index": 1})["projection_jitter"]
        assert j0 != j1


# --- 校验 / 审计 / 降级 / 指标 / 确定性 ------------------------------------

class TestValidation:
    def test_valid_config(self):
        out = mod.validation({"exposure": 1 / 120, "frame_time": 1 / 60,
                              "shutter_offset": -0.5, "max_trail_px": 100})
        assert out["valid"] is True and not out["errors"]

    def test_invalid_and_warning(self):
        out = mod.validation({"exposure": -1, "frame_time": 0,
                              "shutter_offset": -2, "max_trail_px": 0})
        assert out["valid"] is False
        assert len(out["errors"]) == 4
        warn_cfg = mod.validation({"exposure": 1 / 30, "frame_time": 1 / 60})
        assert warn_cfg["warnings"] and warn_cfg["valid"] is True


class TestAudit:
    def test_healthy_snapshot(self):
        assert mod.audit({})["healthy"] is True

    def test_all_six_pathologies(self):
        out = mod.audit({
            "shutter_decoupled": False,
            "motion_composed": False,
            "history_separated": False,
            "skinned_motion_updated": False,
            "results_aligned": False,
            "trail_capped": False,
        })
        assert out["issue_count"] == 6 and not out["healthy"]
        fixes = {issue["id"] for issue in out["issues"]}
        assert fixes == {
            "coupled_shutter", "separate_camera_sampling", "shared_history",
            "stale_skinned_vectors", "late_async_result", "unbounded_trail",
        }


class TestFallbackAndMetrics:
    def test_fallback_safe_config(self):
        out = mod.fallback({"reason": "gpu_reset"})
        assert out["degraded"] is True
        assert out["config"]["max_trail_px"] > 0
        assert out["config"]["exposure"] <= out["config"]["frame_time"]

    def test_metrics_rates(self):
        out = mod.metrics({"frames": [
            {"dropped_frames": 1, "trail_magnitude": 200, "max_trail_px": 100,
             "status": "discarded", "jitter": (0.25, -0.17)},
            {"dropped_frames": 0, "trail_magnitude": 10, "max_trail_px": 100,
             "status": "applied", "jitter": (-0.25, 0.17)},
        ]})
        assert out["total_dropped_frames"] == 1
        assert out["drop_rate"] == pytest.approx(1 / 3)
        assert out["trail_over_limit_rate"] == pytest.approx(0.5)
        assert out["result_discard_rate"] == pytest.approx(0.5)
        assert out["jitter_rms"] > 0


class TestDeterminism:
    def test_repeated_runs_identical(self):
        snapshot = {
            "exposure": 1 / 120, "frame_time": 1 / 60,
            "translation": (3.0, 4.0), "rotation": 0.01,
            "vertices": [{"id": "a", "pos": (2.0, 0.0)}],
            "prev_vertices": [{"id": "a", "pos": (0.0, 0.0)}],
            "frame_index": 3,
        }
        out = mod.determinism({"runs": 3, **snapshot})
        assert out["deterministic"] is True and out["runs"] == 3
        assert out["run_a"] == out["run_b"] and not out["mismatches"]
