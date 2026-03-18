import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldCheck, Users, CheckCircle2, XCircle, Trash2,
  LogOut, Loader2, RefreshCw, Clock, AlertTriangle,
} from "lucide-react";
import { useLocation } from "wouter";

interface UserRow {
  id: number;
  email: string;
  is_approved: boolean;
  created_at: string;
}

function adminFetch(path: string, options: RequestInit = {}) {
  const token = localStorage.getItem("admin_token") ?? "";
  return fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(options.headers ?? {}),
    },
  });
}

export default function AdminDashboard() {
  const [, navigate] = useLocation();
  const [users,   setUsers]   = useState<UserRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");
  const [acting,  setActing]  = useState<number | null>(null); // id of row currently being mutated

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await adminFetch("/api/admin/users");
      if (res.status === 401) { navigate("/admin"); return; }
      const data = await res.json();
      setUsers(data.users ?? []);
    } catch {
      setError("Failed to load users.");
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  useEffect(() => {
    if (!localStorage.getItem("admin_token")) { navigate("/admin"); return; }
    fetchUsers();
  }, [fetchUsers, navigate]);

  const approve = async (id: number) => {
    setActing(id);
    await adminFetch(`/api/admin/approve-user/${id}`, { method: "PUT" });
    await fetchUsers();
    setActing(null);
  };

  const reject = async (id: number) => {
    setActing(id);
    await adminFetch(`/api/admin/reject-user/${id}`, { method: "PUT" });
    await fetchUsers();
    setActing(null);
  };

  const deleteUser = async (id: number) => {
    if (!confirm("Permanently delete this user and their face data?")) return;
    setActing(id);
    await adminFetch(`/api/admin/delete-user/${id}`, { method: "DELETE" });
    await fetchUsers();
    setActing(null);
  };

  const logout = async () => {
    await adminFetch("/api/admin/logout", { method: "POST" });
    localStorage.removeItem("admin_token");
    navigate("/admin");
  };

  const pending  = users.filter(u => !u.is_approved).length;
  const approved = users.filter(u =>  u.is_approved).length;

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-[40%] h-[40%] rounded-full bg-violet-600/4 blur-[120px]" />
        <div className="absolute bottom-0 right-0 w-[35%] h-[35%] rounded-full bg-indigo-600/4 blur-[100px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-white/8 bg-black/30 backdrop-blur-md">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-violet-500/15 border border-violet-500/30 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <h1 className="text-white font-bold text-lg leading-none">Admin Panel</h1>
              <p className="text-gray-500 text-xs mt-0.5">AuraAuth Management Console</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={fetchUsers}
              className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/8 transition-all"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={logout}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-gray-300 hover:text-white bg-white/5 hover:bg-white/10 border border-white/10 transition-all"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      </header>

      <main className="relative z-10 container mx-auto px-6 py-8">

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          {[
            { label: "Total Users",      value: users.length, color: "indigo",  icon: Users },
            { label: "Approved",          value: approved,     color: "green",   icon: CheckCircle2 },
            { label: "Pending Approval",  value: pending,      color: "yellow",  icon: Clock },
          ].map(({ label, value, color, icon: Icon }) => (
            <div key={label} className={`glass-panel rounded-2xl p-5 border-white/10`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">{label}</p>
                  <p className="text-3xl font-bold text-white mt-1">{value}</p>
                </div>
                <div className={`w-11 h-11 rounded-xl flex items-center justify-center ${
                  color === "green"  ? "bg-green-500/15"  :
                  color === "yellow" ? "bg-yellow-500/15" :
                  "bg-indigo-500/15"
                }`}>
                  <Icon className={`w-5 h-5 ${
                    color === "green"  ? "text-green-400"  :
                    color === "yellow" ? "text-yellow-400" :
                    "text-indigo-400"
                  }`} />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Pending approval banner */}
        {pending > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
            className="mb-6 flex items-center gap-3 px-5 py-3.5 rounded-xl bg-yellow-500/10 border border-yellow-500/25 text-yellow-300 text-sm"
          >
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            <span><strong>{pending}</strong> user{pending !== 1 ? "s" : ""} waiting for your approval.</span>
          </motion.div>
        )}

        {/* Users table */}
        <div className="glass-panel rounded-3xl border-white/10 overflow-hidden">
          <div className="px-6 py-4 border-b border-white/8 flex items-center justify-between">
            <h2 className="text-white font-bold flex items-center gap-2">
              <Users className="w-4 h-4 text-indigo-400" />
              Registered Users
            </h2>
            <span className="text-xs text-gray-500">{users.length} total</span>
          </div>

          {loading ? (
            <div className="py-20 flex flex-col items-center gap-3 text-gray-400">
              <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
              <p className="text-sm">Loading users…</p>
            </div>
          ) : error ? (
            <div className="py-16 text-center">
              <XCircle className="w-10 h-10 text-red-400 mx-auto mb-3" />
              <p className="text-gray-400 text-sm">{error}</p>
            </div>
          ) : users.length === 0 ? (
            <div className="py-16 text-center">
              <Users className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400 text-sm">No users registered yet.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/8">
                    <th className="text-left px-6 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">User</th>
                    <th className="text-left px-6 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">Registered</th>
                    <th className="text-left px-6 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                    <th className="text-right px-6 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <AnimatePresence initial={false}>
                    {users.map((user, i) => (
                      <motion.tr
                        key={user.id}
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ delay: i * 0.03 }}
                        className="border-b border-white/5 last:border-0 hover:bg-white/3 transition-colors"
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-indigo-500/15 border border-indigo-500/25 flex items-center justify-center flex-shrink-0">
                              <span className="text-xs font-bold text-indigo-300 uppercase">
                                {user.email[0]}
                              </span>
                            </div>
                            <span className="text-sm text-white font-medium">{user.email}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-400">
                          {new Date(user.created_at).toLocaleDateString("en-US", {
                            year: "numeric", month: "short", day: "numeric",
                          })}
                        </td>
                        <td className="px-6 py-4">
                          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                            user.is_approved
                              ? "bg-green-500/15 text-green-400 border border-green-500/25"
                              : "bg-yellow-500/15 text-yellow-400 border border-yellow-500/25"
                          }`}>
                            {user.is_approved ? (
                              <><CheckCircle2 className="w-3 h-3" />Approved</>
                            ) : (
                              <><Clock className="w-3 h-3" />Pending</>
                            )}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center justify-end gap-2">
                            {acting === user.id ? (
                              <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                            ) : (
                              <>
                                {!user.is_approved ? (
                                  <button
                                    onClick={() => approve(user.id)}
                                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-green-300 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 transition-all"
                                  >
                                    <CheckCircle2 className="w-3.5 h-3.5" />
                                    Approve
                                  </button>
                                ) : (
                                  <button
                                    onClick={() => reject(user.id)}
                                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-yellow-300 bg-yellow-500/10 hover:bg-yellow-500/20 border border-yellow-500/20 transition-all"
                                  >
                                    <XCircle className="w-3.5 h-3.5" />
                                    Revoke
                                  </button>
                                )}
                                <button
                                  onClick={() => deleteUser(user.id)}
                                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-red-300 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 transition-all"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                  Delete
                                </button>
                              </>
                            )}
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
